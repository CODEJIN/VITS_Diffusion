from argparse import Namespace
import torch
import numpy as np
import math
from numba import jit
from typing import Optional, List, Dict, Tuple, Union

from .Diffusion import Difussion
from .Layer import Linear, Conv1d, Lambda
from .monotonic_align import maximum_path

class VITS_Diff(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters
        
        self.encoder = Encoder(self.hp)
        
        self.emotion_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Emotions,
            embedding_dim= self.hp.Encoder.Size,
            )
        torch.nn.init.xavier_uniform_(self.emotion_embedding.weight)

        self.ge2e = Linear(
            in_features= self.hp.GE2E.Size,
            out_features= self.hp.Encoder.Size,
            bias= True,
            w_init_gain= 'linear'
            )

        self.variance_predictor_block = Variance_Predictor_Block(self.hp)

        self.posterior_encoder = Posterior_Encoder(self.hp)
        self.flow = Flow(self.hp)

        self.diffusion = Difussion(self.hp)

        self.segment = Segment()

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        ge2es: torch.Tensor,
        emotions: torch.Tensor,
        features: torch.FloatTensor= None,
        feature_lengths: torch.Tensor= None,
        log_f0s: torch.FloatTensor= None,
        energies: torch.FloatTensor= None,
        audios: torch.FloatTensor= None,
        length_scales: Union[float, List[float], torch.Tensor]= 1.0,
        log_f0_scales: Union[float, List[float], torch.Tensor]= 0.0,
        energy_scales: Union[float, List[float], torch.Tensor]= 0.0
        ):

        if not features is None:
            return self.Train(
                tokens= tokens,
                token_lengths= token_lengths,
                ge2es= ge2es,
                emotions= emotions,
                features= features,
                feature_lengths= feature_lengths,
                log_f0s= log_f0s,
                energies= energies,
                audios= audios,
                )
        else:   # inference
            return self.Inference(
                tokens= tokens,
                token_lengths= token_lengths,
                ge2es= ge2es,
                emotions= emotions,
                length_scales= length_scales,
                log_f0_scales= log_f0_scales,
                energy_scales= energy_scales,
                )

    def Train(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        ge2es: torch.Tensor,
        emotions: torch.Tensor,
        features: torch.FloatTensor,
        feature_lengths: torch.Tensor,
        log_f0s: torch.FloatTensor,
        energies: torch.FloatTensor,
        audios: torch.FloatTensor,
        ):
        encodings, means_p, log_stds_p, token_masks = self.encoder(tokens, token_lengths)   # [Batch, Enc_d, Token_t], [Batch, Enc_d, Token_t], [Batch, Enc_d, Token_t], [Batch, 1, Token_t]
        conditions = self.ge2e(ge2es) + self.emotion_embedding(emotions)    # [Batch, Enc_d]
        posterior_encodings, means_q, log_stds_q, feature_masks = self.posterior_encoder(features, feature_lengths, conditions)  # [Batch, Enc_d, Feature_t], [Batch, Enc_d, Feature_t], [Batch, Enc_d, Feature_t], [Batch, 1, Feature_t]
        posterior_encodings_p = self.flow(posterior_encodings, conditions, feature_masks)   # [Batch, Enc_d, Feature_t]

        with torch.no_grad():
            # negative cross-entropy
            stds_p_sq_r = torch.exp(-2 * log_stds_p) # [Batch, Enc_d, Token_t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - log_stds_p, [1], keepdim=True) # [Batch, 1, Token_t]
            neg_cent2 = torch.matmul(-0.5 * (posterior_encodings_p ** 2).transpose(1, 2), stds_p_sq_r) # [Batch, Feature_t, Enc_d] x [Batch, Enc_d, Token_t] -> [Batch, Feature_t, Token_t]
            neg_cent3 = torch.matmul(posterior_encodings_p.transpose(1, 2), (means_p * stds_p_sq_r)) # [Batch, Feature_t, Enc_d] x [b, Enc_d, Token_t] -> [Batch, Feature_t, Token_t]
            neg_cent4 = torch.sum(-0.5 * (means_p ** 2) * stds_p_sq_r, [1], keepdim=True) # [Batch, 1, Token_t]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4    # [Batch, Feature_t, Token_t]

            attention_masks = token_masks * feature_masks.permute(0, 2, 1)  # [Batch, 1, Token_t] x [Batch, Feature_t, 1] -> [Batch, Feature_t, Token_t]
            attentions = maximum_path(neg_cent, attention_masks).detach()
  
        duration_targets = attentions.sum(dim= 1).long()    # [Batch, Token_t]
        log_f0_targets = (log_f0s.unsqueeze(1) @ attentions).squeeze(1) / duration_targets     # [Batch, Token_t]
        energy_targets = (energies.unsqueeze(1) @ attentions).squeeze(1) / duration_targets    # [Batch, Token_t]
        log_f0_targets = log_f0_targets.nan_to_num(-5.0)    # zero duration fix
        energy_targets = energy_targets.nan_to_num(-1.5)    # zero duration fix

        means_p, log_stds_p, log_duration_predictions, log_f0_predictions, energy_predictions = self.variance_predictor_block(
            encodings= encodings,
            means_p= means_p,
            log_std_p= log_stds_p,
            encoding_masks= token_masks,
            durations= duration_targets,
            log_f0s= log_f0_targets,
            energies= energy_targets,
            )   # [Batch, Enc_d, Feature_t]

        posterior_encodings_slice, offsets = self.segment(
            patterns= posterior_encodings.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            lengths= feature_lengths
            )
        posterior_encodings_slice = posterior_encodings_slice.permute(0, 2, 1)
        audios_slice, _ = self.segment(
            patterns= audios,
            segment_size= self.hp.Train.Segment_Size * self.hp.Sound.Frame_Shift,
            offsets= offsets * self.hp.Sound.Frame_Shift
            )

        predictions, noises, epsilons = self.diffusion(
            encodings= posterior_encodings_slice,
            audios= audios_slice
            )

        '''
        predictions: None
        noises: [Batch, Audio_t]
        epsilons: [Batch, Audio_t]
        durations_targets: [Batch, Token_t]
        log_f0_targets: [Batch, Token_t]
        energy_targets: [Batch, Token_t]
        log_duration_predictions: [Batch, Token_t]
        log_f0_predictions: [Batch, Token_t]
        energy_predictions: [Batch, Token_t]
        encodings: [Batch, Enc_d, Token_t]  Not using
        means_p: [Batch, Enc_d, Token_t]
        log_stds_p: [Batch, Enc_d, Token_t]
        posterior_encodings: [Batch, Enc_d, Feature_t]
        means_q: [Batch, Enc_d, Feature_t]
        log_stds_q: [Batch, Enc_d, Feature_t]
        posterior_encodings_p: [Batch, Enc_d, Feature_t]
        '''

        return \
            predictions, noises, epsilons, \
            duration_targets, log_f0_targets, energy_targets, \
            log_duration_predictions, log_f0_predictions, energy_predictions, \
            encodings, means_p, log_stds_p, \
            posterior_encodings, means_q, log_stds_q, \
            posterior_encodings_p

    def Inference(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        ge2es: torch.Tensor,
        emotions: torch.Tensor,
        length_scales: Union[float, List[float], torch.Tensor]= 1.0,
        log_f0_scales: Union[float, List[float], torch.Tensor]= 0.0,
        energy_scales: Union[float, List[float], torch.Tensor]= 0.0,
        noise_scales: float= 1.0
        ):
        length_scales = self.Scale_to_Tensor(tokens= tokens, scale= length_scales)
        log_f0_scales = self.Scale_to_Tensor(tokens= tokens, scale= log_f0_scales)
        energy_scales = self.Scale_to_Tensor(tokens= tokens, scale= energy_scales)

        encodings, means_p, log_stds_p, token_masks = self.encoder(tokens, token_lengths)   # [Batch, Enc_d, Token_t], [Batch, Enc_d, Token_t], [Batch, Enc_d, Token_t], [Batch, 1, Token_t]
        conditions = self.ge2e(ge2es) + self.emotion_embedding(emotions)    # [Batch, Enc_d]
        
        means_p, log_stds_p, log_duration_predictions, log_f0_predictions, energy_predictions = self.variance_predictor_block(
            encodings= encodings,
            means_p= means_p,
            log_std_p= log_stds_p,
            encoding_masks= token_masks,
            length_scales= length_scales,
            log_f0_scales= log_f0_scales,
            energy_scales= energy_scales,
            )   # [Batch, Enc_d, Feature_t], [Batch, Enc_d, Feature_t], [Batch, Token_t], [Batch, Token_t], [Batch, Token_t]

        feature_masks = (~Mask_Generate(
            lengths= ((torch.exp(log_duration_predictions) - 1).ceil().clip(0, 50) * length_scales).long().sum(dim= 1)
            )).unsqueeze(1).float()  # [Batch, 1, Feature_t]

        posterior_encodings_p = means_p + torch.randn_like(means_p) * torch.exp(log_stds_p) * noise_scales  # [Batch, Enc_d, Feature_t]
        posterior_encodings = self.flow(posterior_encodings_p, conditions, feature_masks, reverse= True)   # [Batch, Enc_d, Feature_t]

        predictions, noises, epsilons = self.diffusion(
            encodings= posterior_encodings
            )   # [Batch, Audio_t], None, None

        '''
        predictions: [Batch, Audio_t]
        noises: None
        epsilons: None
        durations_targets: None
        log_f0_targets: None
        energy_targets: None
        log_duration_predictions: [Batch, Token_t]
        log_f0_predictions: [Batch, Token_t]
        energy_predictions: [Batch, Token_t]
        encodings: [Batch, Enc_d, Token_t]  Not using
        means_p: [Batch, Enc_d, Token_t]
        log_stds_p: [Batch, Enc_d, Token_t]
        posterior_encodings: [Batch, Enc_d, Feature_t]
        means_q: None
        log_stds_q: None
        posterior_encodings_p: [Batch, Enc_d, Feature_t]
        '''

        return \
            predictions, noises, epsilons, \
            None, None, None, \
            log_duration_predictions, log_f0_predictions, energy_predictions, \
            encodings, means_p, log_stds_p, \
            posterior_encodings, None, None, \
            posterior_encodings_p, None

    def Scale_to_Tensor(
        self,
        tokens: torch.Tensor,
        scale: Union[float, List[float], torch.Tensor]
        ):
        if isinstance(scale, float):
            scale = torch.FloatTensor([scale,]).unsqueeze(0).expand(tokens.size(0), tokens.size(1))
        elif isinstance(scale, list):
            if len(scale) != tokens.size(0):
                raise ValueError(f'When scale is a list, the length must be same to the batch size: {len(scale)} != {tokens.size(0)}')
            scale = torch.FloatTensor(scale).unsqueeze(1).expand(tokens.size(0), tokens.size(1))
        elif isinstance(scale, torch.Tensor):
            if scale.ndim != 2:
                raise ValueError('When scale is a tensor, ndim must be 2.')
            elif scale.size(0) != tokens.size(0):
                raise ValueError(f'When scale is a tensor, the dimension 0 of tensor must be same to the batch size: {scale.size(0)} != {tokens.size(0)}')
            elif scale.size(1) != tokens.size(1):
                raise ValueError(f'When scale is a tensor, the dimension 1 of tensor must be same to the token length: {scale.size(1)} != {tokens.size(1)}')

        return scale.to(tokens.device)

class Encoder(torch.nn.Module): 
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters
        assert self.hp.Encoder.Size % 2 == 0, 'The LSTM size of text encoder must be a even number.'

        self.embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size,
            )
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        
        self.conv = torch.nn.Sequential()
        for index in range(self.hp.Encoder.Conv.Stack):
            self.conv.add_module('Conv_{}'.format(index), Conv1d(
                in_channels= self.hp.Encoder.Size,
                out_channels= self.hp.Encoder.Size,
                kernel_size= self.hp.Encoder.Conv.Kernel_Size,
                padding= (self.hp.Encoder.Conv.Kernel_Size - 1) // 2,
                bias= False,
                w_init_gain= 'relu'
                ))
            self.conv.add_module('Mish_{}'.format(index), torch.nn.Mish())
            self.conv.add_module('BatchNorm_{}'.format(index), torch.nn.BatchNorm1d(
                num_features= self.hp.Encoder.Size
                ))
            self.conv.add_module('Dropout_{}'.format(index), torch.nn.Dropout(p= self.hp.Encoder.Conv.Dropout_Rate))

        self.lstm = torch.nn.LSTM(
            input_size= self.hp.Encoder.Size,
            hidden_size= self.hp.Encoder.Size // 2,
            num_layers= self.hp.Encoder.LSTM.Stack,
            bidirectional= True
            )
        self.lstm_dropout = torch.nn.Dropout(
            p= self.hp.Encoder.LSTM.Dropout_Rate
            )

        self.projection = torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size * 2,
            kernel_size= 1,
            )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor):
        '''
        x: [Batch, Time]
        lengths: [Batch]
        '''
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(tokens[0]).sum())).unsqueeze(1).float()

        tokens = self.embedding(tokens).permute(0, 2, 1)     # [Batch, Dim, Time]
        tokens = self.conv(tokens)    # [Batch, Dim, Time]

        unpacked_length = tokens.size(2)
        tokens = torch.nn.utils.rnn.pack_padded_sequence(
            tokens.permute(2, 0, 1),
            lengths.cpu().numpy(),
            enforce_sorted= False
            )
        tokens = self.lstm(tokens)[0]
        tokens = torch.nn.utils.rnn.pad_packed_sequence(
            sequence= tokens,
            total_length= unpacked_length
            )[0].permute(1, 2, 0) * masks   # [Batch, Dim, Time]

        tokens = self.lstm_dropout(tokens)
        means, log_stds = (self.projection(tokens) * masks).chunk(chunks= 2, dim= 1)

        return tokens, means, log_stds, masks


class Flow(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.flow_blocks = torch.nn.ModuleList()
        for index in range(self.hp.Flow.Flow_Stack):
            self.flow_blocks.append(Flow_Block(self.hp))

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor,
        masks: torch.Tensor,
        reverse: bool= False
        ) -> torch.Tensor:
        blocks = self.flow_blocks if not reverse else reversed(self.flow_blocks)
        for block in blocks:
            x = block(x, conditions, masks, reverse)

        return x

class Flow_Block(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv1d(
            in_channels= self.hp.Encoder.Size // 2,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1
            )
        self.wavenet = WaveNet(
            channels= self.hp.Encoder.Size,
            condition_channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Flow.WaveNet.Kernel_Size,
            dilation_rate= self.hp.Flow.WaveNet.Dilation_Rate,
            stack= self.hp.Flow.WaveNet.Stack,
            dropout_rate= self.hp.Flow.WaveNet.Dropout_Rate
            )
        self.postnet = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size // 2,
            kernel_size= 1
            )

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor,
        masks: torch.Tensor,
        reverse: bool= False
        ) -> torch.Tensor:
        if reverse:
            x= x.flip(dims= (1,))

        x_0, x_1 = x.chunk(chunks= 2, dim= 1)
        x = self.prenet(x_0)
        x = self.wavenet(x, conditions, masks)
        x = self.postnet(x)

        if not reverse:
            return torch.cat([x_0, x_1 + x], dim= 1).flip(dims= (1,))
        else:
            return torch.cat([x_0, x_1 - x], dim= 1)

class WaveNet(torch.nn.Module): 
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        kernel_size: int,
        dilation_rate: int,
        stack: int,
        dropout_rate: float
        ):
        super().__init__()
        self.in_blocks = torch.nn.ModuleList()
        for index in range(stack):
            dilation = dilation_rate ** index
            padding = ((kernel_size - 1) * dilation) // 2
            self.in_blocks.append(torch.nn.utils.weight_norm(Conv1d(
                in_channels= channels,
                out_channels= channels * 2,
                kernel_size= kernel_size,
                dilation= dilation,
                padding= padding
                )))

        self.gated_activation = Gated_Activation()
        self.dropout = torch.nn.Dropout(p= dropout_rate)

        self.resisual_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv1d(
                    in_channels= channels,
                    out_channels= channels,
                    kernel_size= 1
                    )),
                )            
            for index in range(stack)
            ])
        self.skip_blocks = torch.nn.ModuleList([
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= channels,
                out_channels= channels,
                kernel_size= 1
                ))
            for index in range(stack)
            ])

        self.condition = torch.nn.Sequential(
            Lambda(lambda x: x.unsqueeze(2)),
            torch.nn.utils.weight_norm(Conv1d(
                in_channels= condition_channels,
                out_channels= stack * channels * 2,
                kernel_size= 1,
                w_init_gain= 'gate'
                )),
            Lambda(lambda x: x.view(
                x.size(0),
                stack,
                channels * 2, 1
                ).permute(1, 0, 2, 3)),
            )

    def forward(
        self,
        x: torch.Tensor,
        condtions: torch.Tensor,
        masks: torch.Tensor
        ) -> torch.Tensor:
        condition_stacks = self.condition(condtions)    # [Stack, Batch, Dim * 2, 1]

        skips_list = []
        for index, (in_block, residual_block, skip_block, conditions) in enumerate(zip(
            self.in_blocks, self.resisual_blocks, self.skip_blocks, condition_stacks
            )):
            x_ins = in_block(x)
            x_ins = self.gated_activation(x_ins, conditions)
            x_ins = self.dropout(x_ins)
            
            skips_list.append(skip_block(x_ins))
            if index < len(self.in_blocks) - 1:
                x = (x + residual_block(x_ins)) * masks
            
        return torch.stack(skips_list, dim= 1).sum(dim= 1) * masks

class Gated_Activation(torch.nn.Module): 
    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor
        ):
        x_tanh, x_sigmoid = (x_a + x_b).chunk(chunks= 2, dim= 1)
        return x_tanh.tanh() * x_sigmoid.sigmoid()


class Posterior_Encoder(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        feature_size = self.hp.Sound.N_FFT // 2 + 1

        self.prenet = Conv1d(
            in_channels = feature_size,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1
            )
        self.wavenet = WaveNet(
            channels= self.hp.Encoder.Size,
            condition_channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Posterior_Encoder.WaveNet.Kernel_Size,
            dilation_rate= self.hp.Posterior_Encoder.WaveNet.Dilation_Rate,
            stack= self.hp.Posterior_Encoder.WaveNet.Stack,
            dropout_rate= self.hp.Posterior_Encoder.WaveNet.Dropout_Rate
            )
        self.projection = Conv1d(
            in_channels = self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size * 2,
            kernel_size= 1
            )

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        conditions: torch.Tensor,
        ) -> torch.Tensor:
        masks = (~Mask_Generate(lengths= feature_lengths, max_length= torch.ones_like(features[0, 0]).sum())).unsqueeze(1).float()

        features = self.prenet(features)
        features = self.wavenet(features, conditions, masks)
        means, log_stds = (self.projection(features) * masks).chunk(chunks= 2, dim= 1)
        posterior_encodings = means + torch.randn_like(log_stds) * masks * log_stds.exp()
 
        return posterior_encodings, means, log_stds, masks


class Variance_Predictor_Block(torch.nn.Module): 
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.duration_predictor = Variance_Predictor(
            in_channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Variance.Duration.Channels,
            out_channels= 1,
            conv_kernel_size= self.hp.Variance.Duration.Kernel_Size,
            dropout_rate= self.hp.Variance.Duration.Dropout_Rate
            )
        self.length_regulator = Length_Regulator()

        self.log_f0_predictor = Variance_Predictor(
            in_channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Variance.Log_F0.Predictor.Channels,
            out_channels= 1,
            conv_kernel_size= self.hp.Variance.Log_F0.Predictor.Kernel_Size,
            dropout_rate= self.hp.Variance.Log_F0.Predictor.Dropout_Rate
            )
        self.log_f0_embedding = Conv1d(
            in_channels= 1,
            out_channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Variance.Log_F0.Kernel_Size,
            padding= (self.hp.Variance.Log_F0.Kernel_Size - 1) // 2,
            bias= True,
            w_init_gain= 'linear'
            )

        self.energy_predictor = Variance_Predictor(
            in_channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Variance.Energy.Predictor.Channels,
            out_channels= 1,
            conv_kernel_size= self.hp.Variance.Energy.Predictor.Kernel_Size,
            dropout_rate= self.hp.Variance.Energy.Predictor.Dropout_Rate
            )
        self.energy_embedding = Conv1d(
            in_channels= 1,
            out_channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Variance.Energy.Kernel_Size,
            padding= (self.hp.Variance.Energy.Kernel_Size - 1) // 2,
            bias= True,
            w_init_gain= 'linear'
            )

    def forward(
        self,
        encodings: torch.Tensor,
        means_p: torch.Tensor,
        log_std_p: torch.Tensor,
        encoding_masks: torch.Tensor,
        durations: torch.Tensor= None,
        log_f0s: torch.Tensor= None,
        energies: torch.Tensor= None,
        length_scales: torch.Tensor= None,
        log_f0_scales: torch.Tensor= None,
        energy_scales: torch.Tensor= None,
        ):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        durations: [Batch, Enc_t]
        '''                
        encodings = encodings.detach()
        log_duration_predictions = self.duration_predictor(encodings).squeeze(1)   # [Batch, Enc_t]
        if durations is None:
            durations = ((torch.exp(log_duration_predictions) - 1).ceil().clip(0, 50) * length_scales).long()
            durations[:, -1] += durations.sum(dim= 1).max() - durations.sum(dim= 1) # Align the sum of lengths

        log_f0_predictions = self.log_f0_predictor(encodings).masked_fill(~encoding_masks.bool(), -5.0)   # [Batch, 1, Enc_t]
        energy_predictions = self.energy_predictor(encodings).masked_fill(~encoding_masks.bool(), -1.5)   # [Batch, 1, Enc_t]

        log_f0s = log_f0s.unsqueeze(1) if not log_f0s is None else (log_f0_predictions + log_f0_scales.unsqueeze(1) * (log_f0_predictions > -5.0)).clip(-5.0, torch.inf)
        energies = energies.unsqueeze(1) if not energies is None else (energy_predictions + energy_scales.unsqueeze(1)).clip(-1.5, torch.inf)

        log_f0s = self.log_f0_embedding(log_f0s) * encoding_masks
        energies = self.energy_embedding(energies) * encoding_masks

        means_p = means_p + log_f0s + energies

        means_p = self.length_regulator(
            encodings= means_p,
            durations= durations
            )
        log_std_p = self.length_regulator(
            encodings= log_std_p,
            durations= durations
            )
        
        return means_p, log_std_p, log_duration_predictions, log_f0_predictions.squeeze(1), energy_predictions.squeeze(1)

class Variance_Predictor(torch.nn.Sequential): 
    def __init__(
        self,
        in_channels: int,
        calc_channels: int,
        out_channels: int,
        conv_kernel_size: int,
        dropout_rate: float= 0.1
        ):
        super().__init__()

        self.add_module('Conv_0', Conv1d(
            in_channels= in_channels,
            out_channels= calc_channels,
            kernel_size= conv_kernel_size,
            padding= (conv_kernel_size - 1) // 2,
            bias= True,
            w_init_gain= 'relu'
            ))
        self.add_module('Mish_0', torch.nn.Mish(inplace= False))
        self.add_module('Norm_0', torch.nn.BatchNorm1d(
            num_features= calc_channels,
            ))
        self.add_module('Dropout_0', torch.nn.Dropout(
            p= dropout_rate
            ))

        self.add_module('Conv_1', Conv1d(
            in_channels= calc_channels,
            out_channels= calc_channels,
            kernel_size= conv_kernel_size,
            padding= (conv_kernel_size - 1) // 2,
            bias= True,
            w_init_gain= 'relu'
            ))
        self.add_module('Mish_1', torch.nn.Mish(inplace= False))
        self.add_module('Norm_1', torch.nn.BatchNorm1d(
            num_features= calc_channels,
            ))
        self.add_module('Dropout_1', torch.nn.Dropout(
            p= dropout_rate
            ))

        self.add_module('Projection', Conv1d(
            in_channels= calc_channels,
            out_channels= out_channels,
            kernel_size= 1,
            bias= True,
            w_init_gain= 'linear'
            ))

    def forward(self, x: torch.Tensor):
        '''
        x: [Batch, Dim, Time]
        '''
        return super().forward(x)

class Length_Regulator(torch.nn.Module):
    def forward(
        self,
        encodings= torch.Tensor,
        durations= torch.Tensor
        ):
        durations = torch.cat([
            durations,
            durations.sum(dim= 1).max() - durations.sum(dim= 1, keepdim= True)
            ], dim= 1)
        encodings = torch.cat([encodings, torch.zeros_like(encodings[:, :, -1:])], dim= 2)

        return torch.stack([
            encoding.repeat_interleave(duration, dim= 1)
            for encoding, duration in zip(encodings, durations)
            ], dim= 0)

class Segment(torch.nn.Module):
    def forward(
        self,
        patterns: torch.Tensor,
        segment_size: int,
        lengths: torch.Tensor= None,
        offsets: torch.Tensor= None
        ):
        '''
        patterns: [Batch, Time, ...]
        lengths: [Batch]
        segment_size: an integer scalar    
        '''
        if offsets is None:
            offsets = (torch.rand_like(patterns[:, 0, 0]) * (lengths - segment_size)).long()
        segments = torch.stack([
            pattern[offset:offset + segment_size]
            for pattern, offset in zip(patterns, offsets)
            ], dim= 0)
        
        return segments, offsets


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/soobinseo/Transformer-TTS/blob/master/network.py
class Periodic_Positional_Encoding(torch.nn.Module):
    def __init__(
        self,
        period: int,
        embedding_size: int,
        dropout_rate: float
        ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        pe = torch.zeros(period, embedding_size)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(2, 1)
        self.register_buffer('pe', pe)

        self.alpha = torch.nn.Parameter(
            data= torch.ones(1),
            requires_grad= True
            )

    def forward(self, x):
        '''
        x: [Batch, Dim, Length]
        '''
        x = x + self.alpha * self.get_pe(x, self.pe)
        x = self.dropout(x)

        return x

    @torch.jit.script
    def get_pe(x: torch.Tensor, pe: torch.Tensor):
        pe = pe.repeat(1, 1, math.ceil(x.size(2) / pe.size(2))) 
        return pe[:, :, :x.size(2)]

def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]


def KL_Loss(posterior_encodings_p, log_stds_q, means_p, log_stds_p, feature_masks):
    losses = log_stds_p - log_stds_q - 0.5 + 0.5 * (posterior_encodings_p - means_p).pow(2.0) * (-2.0 * log_stds_p).exp()
    losses = (losses * feature_masks).sum() / feature_masks.sum()

    return losses