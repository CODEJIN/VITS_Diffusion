from argparse import Namespace
import torch
import numpy as np
import math
from numba import jit
from typing import Optional, List, Dict, Tuple, Union

from .Diffusion import Diffusion
from .Layer import Linear, Conv1d, Lambda
from .monotonic_align import maximum_path
from .vits_transforms import piecewise_rational_quadratic_transform

class VITS_Diff(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters
        
        self.encoder = Encoder(self.hp)        
        self.variance_predictor_block = Variance_Predictor_Block(self.hp)
        
        self.stochastic_duration_predictor = Stochastic_Duration_Predictor(self.hp)
        self.length_regulator = Length_Regulator()

        self.diffusion = Diffusion(self.hp)

        self.segment = Segment()
        # self.maximum_path_generator = Maximum_Path_Generator()

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        features: torch.FloatTensor= None,
        feature_lengths: torch.Tensor= None,
        audios: torch.FloatTensor= None,
        length_scales: Union[float, List[float], torch.Tensor]= 1.0,
        ):

        if not features is None:
            return self.Train(
                tokens= tokens,
                token_lengths= token_lengths,
                features= features,
                feature_lengths= feature_lengths,
                audios= audios,
                )
        else:   # inference
            return self.Inference(
                tokens= tokens,
                token_lengths= token_lengths,
                length_scales= length_scales,
                )

    def Train(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        features: torch.FloatTensor,
        feature_lengths: torch.Tensor,
        audios: torch.FloatTensor,
        ):        
        encodings, means_p, log_stds_p, token_masks = self.encoder(tokens, token_lengths)   # [Batch, Feature_d, Token_t], [Batch, Feature_d, Token_t], [Batch, Feature_d, Token_t], [Batch, 1, Token_t]        
        feature_masks = (~Mask_Generate(
            lengths= feature_lengths,
            max_length= torch.ones_like(features[0, 0]).sum()
            )).unsqueeze(1).float()
        
        with torch.no_grad():
            # negative cross-entropy
            stds_p_sq_r = torch.exp(-2 * log_stds_p) # [Batch, Feature_d, Token_t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - log_stds_p, [1], keepdim=True) # [Batch, 1, Token_t]
            neg_cent2 = torch.matmul(-0.5 * (features ** 2).permute(0, 2, 1), stds_p_sq_r) # [Batch, Feature_t, Feature_d] x [Batch, Feature_d, Token_t] -> [Batch, Feature_t, Token_t]
            neg_cent3 = torch.matmul(features.permute(0, 2, 1), (means_p * stds_p_sq_r)) # [Batch, Feature_t, Feature_d] x [b, Feature_d, Token_t] -> [Batch, Feature_t, Token_t]
            neg_cent4 = torch.sum(-0.5 * (means_p ** 2) * stds_p_sq_r, [1], keepdim=True) # [Batch, 1, Token_t]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4    # [Batch, Feature_t, Token_t]

            attention_masks = token_masks * feature_masks.permute(0, 2, 1)  # [Batch, 1, Token_t] x [Batch, Feature_t, 1] -> [Batch, Feature_t, Token_t]
            attentions = maximum_path(neg_cent, attention_masks).detach()
            # attentions = self.maximum_path_generator(neg_cent, attention_masks).detach()

        duration_targets = attentions.sum(dim= 1).long()    # [Batch, Token_t]

        # means_p, log_stds_p, log_duration_predictions = self.variance_predictor_block(
        #     encodings= encodings,
        #     means_p= means_p,
        #     log_std_p= log_stds_p,
        #     durations= duration_targets
        #     )   # [Batch, Feature_d, Feature_t]
        log_duration_predictions = self.stochastic_duration_predictor(
            encodings= encodings,
            masks= token_masks,
            weights= attentions.sum(dim= 1)
            ) / token_masks.sum()
        means_p = means_p @ attentions.permute(0, 2, 1) # [Batch, Feature_d, Token_t] @ [Batch, Token_t, Feature_t] -> [Batch, Feature_d, Feature_t]
        log_stds_p = log_stds_p @ attentions.permute(0, 2, 1)   # [Batch, Feature_d, Token_t] @ [Batch, Token_t, Feature_t] -> [Batch, Feature_d, Feature_t]
        

        means_p_slice, offsets = self.segment(
            patterns= means_p.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            lengths= feature_lengths
            )
        means_p_slice = means_p_slice.permute(0, 2, 1)
        audios_slice, _ = self.segment(
            patterns= audios,
            segment_size= self.hp.Train.Segment_Size * self.hp.Sound.Frame_Shift,
            offsets= offsets * self.hp.Sound.Frame_Shift
            )

        predictions, noises, epsilons = self.diffusion(
            conditions= means_p_slice,
            audios= audios_slice
            )

        '''
        predictions: None
        noises: [Batch, Audio_t]
        epsilons: [Batch, Audio_t]
        durations_targets: [Batch, Token_t]
        log_duration_predictions: [Batch, Token_t]
        encodings: [Batch, Enc_d, Token_t]  Not using
        means_p: [Batch, Enc_d, Token_t]
        log_stds_p: [Batch, Enc_d, Token_t]
        '''

        return \
            predictions, noises, epsilons, \
            duration_targets, log_duration_predictions, \
            encodings, means_p, log_stds_p

    def Inference(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        length_scales: Union[float, List[float], torch.Tensor]= 1.0
        ):
        length_scales = self.Scale_to_Tensor(tokens= tokens, scale= length_scales)

        encodings, means_p, log_stds_p, token_masks = self.encoder(tokens, token_lengths)   # [Batch, Enc_d, Token_t], [Batch, Enc_d, Token_t], [Batch, Enc_d, Token_t], [Batch, 1, Token_t]        
        
        # means_p, log_stds_p, log_duration_predictions = self.variance_predictor_block(
        #     encodings= encodings,
        #     means_p= means_p,
        #     log_std_p= log_stds_p,
        #     length_scales= length_scales,
        #     )   # [Batch, Enc_d, Feature_t], [Batch, Enc_d, Feature_t], [Batch, Token_t], [Batch, Token_t], [Batch, Token_t]
        # feature_masks = (~Mask_Generate(
        #     lengths= ((torch.exp(log_duration_predictions) - 1).ceil().clip(0, 50) * length_scales).long().sum(dim= 1)
        #     )).unsqueeze(1).float()  # [Batch, 1, Feature_t]
        log_duration_predictions = self.stochastic_duration_predictor(
            encodings= encodings,
            masks= token_masks,
            reverse=True
            )
        durations = torch.exp(log_duration_predictions) * token_masks * length_scales
        durations = torch.ceil(durations).long().squeeze(1)
        means_p = self.length_regulator(
            encodings= means_p,
            durations= durations
            )
        log_stds_p = self.length_regulator(
            encodings= log_stds_p,
            durations= durations
            )

        predictions, noises, epsilons = self.diffusion(
            conditions= means_p,
            )   # [Batch, Audio_t], None, None

        '''
        predictions: [Batch, Audio_t]
        noises: None
        epsilons: None
        durations_targets: None
        log_duration_predictions: [Batch, Token_t]
        encodings: [Batch, Enc_d, Token_t]  Not using
        means_p: [Batch, Enc_d, Token_t]
        log_stds_p: [Batch, Enc_d, Token_t]
        '''

        return \
            predictions, noises, epsilons, \
            None, log_duration_predictions, \
            encodings, means_p, log_stds_p

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
        
        feature_size = self.hp.Sound.N_FFT // 2 + 1

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
            out_channels= feature_size * 2,
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

    def forward(
        self,
        encodings: torch.Tensor,
        means_p: torch.Tensor,
        log_std_p: torch.Tensor,
        durations: torch.Tensor= None,
        length_scales: torch.Tensor= None,
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

        means_p = self.length_regulator(
            encodings= means_p,
            durations= durations
            )
        log_std_p = self.length_regulator(
            encodings= log_std_p,
            durations= durations
            )
        
        return means_p, log_std_p, log_duration_predictions

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



class Stochastic_Duration_Predictor(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.encoding_prenet = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1
            )

        self.encoding_dds = Dilated_Depth_Separable_Conv(
            channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Duration_Predictor.DDS.Kernel_Size,
            stack= self.hp.Duration_Predictor.DDS.Stack,
            dropout_rate= self.hp.Duration_Predictor.DDS.Dropout_Rate,
            )
        self.encoding_projection = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1
            )

        self.weight_prenet = Conv1d(
            in_channels= 1,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1
            )
        self.weight_dds = Dilated_Depth_Separable_Conv(
            channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Duration_Predictor.DDS.Kernel_Size,
            stack= self.hp.Duration_Predictor.DDS.Stack,
            dropout_rate= self.hp.Duration_Predictor.DDS.Dropout_Rate,
            )
        self.weight_projection = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1
            )
        self.weight_affine = Affine(2)
        self.weight_conv_flow = torch.nn.ModuleList([
            Conv_Flow(
                in_channels= 2,
                calc_channels= self.hp.Encoder.Size,
                dds_kernel_size= self.hp.Duration_Predictor.DDS.Kernel_Size,
                dds_stack= self.hp.Duration_Predictor.DDS.Stack,
                dds_dropout_rate=  self.hp.Duration_Predictor.DDS.Dropout_Rate,
                bins= self.hp.Duration_Predictor.Weight_Flow.Bins
                )
            for index in range(self.hp.Duration_Predictor.Weight_Flow.Stack)
            ])

        self.affine = Affine(2)
        self.conv_flow = torch.nn.ModuleList([
            Conv_Flow(
                in_channels= 2,
                calc_channels= self.hp.Encoder.Size,
                dds_kernel_size= self.hp.Duration_Predictor.DDS.Kernel_Size,
                dds_stack= self.hp.Duration_Predictor.DDS.Stack,
                dds_dropout_rate=  self.hp.Duration_Predictor.DDS.Dropout_Rate,
                bins= self.hp.Duration_Predictor.Flow.Bins
                )
            for index in range(self.hp.Duration_Predictor.Flow.Stack)
            ])

    def forward(
        self,
        encodings: torch.Tensor,    # [Batch, Enc_d, Token_t]
        masks: torch.Tensor,    # [Batch, 1, Token_t]
        weights: torch.Tensor= None,    # [Batch, Token_t]
        reverse: bool= False,
        noise_scale: float= 1.0
        ) -> torch.Tensor:
        x = self.encoding_prenet(encodings.detach())
        x = self.encoding_dds(x, masks)
        x = self.encoding_projection(x) * masks

        if not reverse:
            weights = weights.unsqueeze(1)
            weights_hidden = self.weight_prenet(weights)
            weights_hidden = self.weight_dds(weights_hidden, masks)
            weights_hidden = self.weight_projection(weights_hidden) * masks
            torch.randn(weights.size(2))

            e_q = torch.randn(weights.size(0), 2, weights.size(2)).to(device=weights.device, dtype=weights.dtype)
            z_q = e_q

            log_det_q_list = []
            z_q, log_det_q = self.weight_affine(
                x= z_q,
                masks= masks,
                reverse= reverse
                )
            log_det_q_list.append(log_det_q)
            for conv_flow in self.weight_conv_flow:
                z_q, log_det_q = conv_flow(
                    x= z_q,
                    conditions= x + weights_hidden,
                    masks= masks,
                    reverse= reverse
                    )
                log_det_q_list.append(log_det_q)

            z_u, z_1 = z_q.chunk(chunks= 2, dim= 1)
            u = z_u.sigmoid() * masks
            z_0 = (weights - u) * masks
            log_det_q_list.append(torch.sum(
                (torch.nn.functional.logsigmoid(z_u) + torch.nn.functional.logsigmoid(-z_u)) * masks,
                dim= (1, 2)
                ))
            log_det_q = torch.stack(log_det_q_list, dim= 1).sum(dim= 1)
            log_q = (-0.5 * (math.log(2 * math.pi) + e_q.pow(2.0)) * masks).sum(dim= (1, 2)) - log_det_q

            log_det_list = []
            z_0 = z_0.clamp_min(1e-5).log() * masks
            z = torch.cat([z_0, z_1], dim= 1)
            log_det_list.append((-z_0).sum(dim= (1, 2)))
            z, log_det = self.affine(
                x= z,
                masks= masks,
                reverse= reverse
                )
            log_det_list.append(log_det)
            for conv_flow in self.conv_flow:
                z, log_det = conv_flow(
                    x= z,
                    conditions= x,
                    masks= masks,
                    reverse= reverse
                    )
                log_det_list.append(log_det)
            log_det = torch.stack(log_det_list, dim= 1).sum(dim= 1)
            nll = (0.5 * (math.log(2 * math.pi) + z.pow(2.0)) * masks).sum(dim= (1, 2)) - log_det

            return nll + log_q
        else:
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale

            for conv_flow in list(reversed(self.conv_flow))[:-1]:
                z, _ = conv_flow(
                    x= z,
                    conditions= x,
                    masks= masks,
                    reverse= reverse
                    )
            z, _ = self.weight_affine(
                x= z,
                masks= masks,
                reverse= reverse
                )
            log_w = z.chunk(chunks= 2, dim= 1)[0]

            return log_w


class Dilated_Depth_Separable_Conv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stack: int,
        dropout_rate: float
        ):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for index in range(stack):
            dilation = kernel_size ** index
            padding = (kernel_size - 1) * dilation // 2
            self.blocks.append(torch.nn.Sequential(
                Conv1d(
                    in_channels= channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    dilation= dilation,
                    padding= padding
                    ),
                torch.nn.Mish(),
                torch.nn.BatchNorm1d(
                    num_features= channels
                    ),
                Conv1d(
                    in_channels= channels,
                    out_channels= channels,
                    kernel_size= 1
                    ),
                torch.nn.Mish(),
                torch.nn.BatchNorm1d(
                    num_features= channels
                    ),
                torch.nn.Dropout(p= dropout_rate)
                ))

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x * masks)
        
        return x * masks

class Affine(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.mean = torch.nn.Parameter(torch.zeros((channels, 1)))
        self.log_std = torch.nn.Parameter(torch.zeros((channels, 1)))

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        reverse: bool= False,
        ) -> torch.Tensor:
        if not reverse:
            x = (self.mean + self.log_std.exp() * x) * masks
            log_dets = (self.log_std * masks).sum(dim= [1, 2]) # [Ch, 1] * [Batch, 1, Time] -> [Batch, Ch, Time] -> [Batch]
            
            return x, log_dets
        else:
            x = (x - self.mean) / self.log_std.exp() * masks
            return x, None

class Conv_Flow(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        calc_channels: int,
        dds_kernel_size: int,
        dds_stack: int,
        dds_dropout_rate: float,
        bins: int,
        tail_bound: float= 5.0
        ):
        super().__init__()
        assert in_channels % 2 == 0, 'in_channels must be an even.'

        self.calc_channels = calc_channels
        self.bins = bins
        self.tail_bound = tail_bound

        self.prenet = Conv1d(
            in_channels= in_channels // 2,
            out_channels= calc_channels,
            kernel_size= 1
            )
        self.dds = Dilated_Depth_Separable_Conv(
            channels= calc_channels,
            kernel_size= dds_kernel_size,
            stack= dds_stack,
            dropout_rate= dds_dropout_rate
            )
        self.projection = torch.nn.Sequential(
            Conv1d(
                in_channels= calc_channels,
                out_channels= in_channels // 2 * (3 * bins - 1),
                kernel_size= 1
                ),            
            Lambda(lambda x: x.view(x.size(0), in_channels // 2, 3 * bins - 1, x.size(2)).permute(0, 1, 3, 2)),
            )   # [Batch, Half_Ch, Time, 3 * Bin - 1]        
        torch.nn.init.zeros_(self.projection[0].weight)
        torch.nn.init.zeros_(self.projection[0].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor,
        masks: torch.Tensor,
        reverse: bool= False
        ):        
        if reverse:
            x= x.flip(dims= (1,))

        x_0, x_1 = x.chunk(chunks= 2, dim= 1)

        x_hidden = x_0
        x_hidden = self.prenet(x_hidden)
        x_hidden = self.dds(x_hidden + conditions, masks)

        x_hidden = self.projection(x_hidden) * masks.unsqueeze(3)
        weights, heights, derivates = x_hidden.split(split_size= [self.bins, self.bins, self.bins - 1], dim= 3)
        weights = weights /math.sqrt(self.calc_channels) 
        heights = heights /math.sqrt(self.calc_channels)

        x_1, log_abs_det = piecewise_rational_quadratic_transform(
            inputs= x_1,
            unnormalized_widths= weights,
            unnormalized_heights= heights,
            unnormalized_derivatives= derivates,
            inverse= reverse,
            tails= 'linear', 
            tail_bound= self.tail_bound
            )

        x = torch.cat([x_0, x_1], dim= 1) * masks
        if not reverse:
            x = x.flip(dims= (1,))
        log_det = (log_abs_det * masks).sum(dim= [1, 2])

        return x, log_det




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

class Maximum_Path_Generator(torch.nn.Module):
    def forward(self, log_p, mask):
        '''
        x: [Batch, Feature_t, Token_t]
        mask: [Batch, Feature_t, Token_t]
        '''
        log_p *= mask
        device, dtype = log_p.device, log_p.dtype
        log_p = log_p.data.cpu().numpy().astype(np.float32)
        mask = mask.data.cpu().numpy()

        token_lengths = mask.sum(axis= 2)[:, 0].astype('int32')   # [Batch]
        feature_lengths = mask.sum(axis= 1)[:, 0].astype('int32')   # [Batch]

        paths = self.calc_paths(log_p, token_lengths, feature_lengths)

        return torch.from_numpy(paths).to(device= device, dtype= dtype)

    def calc_paths(self, log_p, token_lengths, feature_lengths):
        return np.stack([
            Maximum_Path_Generator.calc_path(x, token_length, feature_length)
            for x, token_length, feature_length in zip(log_p, token_lengths, feature_lengths)
            ], axis= 0)

    @staticmethod
    @jit(nopython=True)
    def calc_path(x, token_length, feature_length):
        path = np.zeros_like(x, dtype= np.int32)
        for feature_index in range(feature_length):
            for token_index in range(max(0, token_length + feature_index - feature_length), min(token_length, feature_index + 1)):
                if feature_index == token_index:
                    current_q = -1e+9
                else:
                    current_q = x[feature_index - 1, token_index]   # Stayed current token
                if token_index == 0:
                    if feature_index == 0:
                        prev_q = 0.0
                    else:
                        prev_q = -1e+9
                else:
                    prev_q = x[feature_index - 1, token_index - 1]  # Moved to next token
            x[feature_index, token_index] = x[feature_index, token_index] + max(prev_q, current_q)

        token_index = token_length - 1
        for feature_index in range(feature_length - 1, -1, -1):
            path[feature_index, token_index] = 1
            if token_index != 0 and token_index == feature_index or x[feature_index - 1, token_index] < x[feature_index - 1, token_index - 1]:
                token_index = token_index - 1

        return path

def KL_Loss(features, means_p, log_stds_p, feature_masks):
    losses = log_stds_p.sum() + 0.5 * ((-2.0 * log_stds_p).exp() * (features - means_p).pow(2.0)).sum()
    losses = losses / (torch.ones_like(features) * feature_masks).sum()
    losses = losses + 0.5 * math.log(2.0 * math.pi)
    
    return losses