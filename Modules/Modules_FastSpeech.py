from argparse import Namespace
import torch
import numpy as np
import math
from numba import jit
from typing import Optional, List, Dict, Tuple, Union

from .Layer import Linear, Conv1d, Lambda

class Model_Test(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters
        
        self.encoder = Encoder(self.hp)
        self.variance_predictor_block = Variance_Predictor_Block(self.hp)
        self.decoder = Decoder(self.hp)
        self.segment = Segment()

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        features: torch.FloatTensor= None,
        feature_lengths: torch.Tensor= None,
        durations: torch.Tensor= None
        ):
        if not features is None and not feature_lengths is None:    # train
            return self.Train(
                tokens= tokens,
                token_lengths= token_lengths,
                features= features,
                feature_lengths= feature_lengths,
                durations= durations
                )
        else:   #  inference
            return self.Inference(
                tokens= tokens,
                token_lengths= token_lengths
                )

    def Train(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        features: torch.FloatTensor,
        feature_lengths: torch.Tensor,
        durations: torch.Tensor
        ):
        encodings, token_masks = self.encoder(tokens, token_lengths)   # [Batch, Enc_d, Token_t], [Batch, Enc_d, Token_t]
        encodings, log_duration_predictions = self.variance_predictor_block(
            encodings= encodings,
            durations= durations
            )

        encodings_slice, offsets = self.segment(
            patterns= encodings.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            lengths= feature_lengths
            )
        encodings_slice = encodings_slice.permute(0, 2, 1)
        features_slice, _ = self.segment(
            patterns= features.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            offsets= offsets
            )
        features_slice = features_slice.permute(0, 2, 1)
        predictions_slice, feature_masks = self.decoder(
            encodings= encodings_slice,
            lengths= torch.full_like(feature_lengths, fill_value= encodings_slice.size(2))
            )

        return predictions_slice, features_slice, log_duration_predictions

    def Inference(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        ):
        encodings, token_masks = self.encoder(tokens, token_lengths)   # [Batch, Enc_d, Token_t], [Batch, Enc_d, Token_t]
        encodings, log_duration_predictions = self.variance_predictor_block(
            encodings= encodings
            )

        predictions, feature_masks = self.decoder(
            encodings= encodings,
            lengths= ((log_duration_predictions.exp() - 1).clip(0, 50).ceil()).sum(dim= 1).long()
            )

        return predictions, None, log_duration_predictions


class Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        if self.hp.Feature_Type == 'Mel':
            feature_size = self.hp.Sound.Mel_Dim
        elif self.hp.Feature_Type == 'Spectrogram':
            feature_size = self.hp.Sound.N_FFT // 2 + 1

        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(
                num_embeddings= self.hp.Tokens,
                embedding_dim= self.hp.Encoder.Size,
                ),
            Lambda(lambda x: x.permute(0, 2, 1))
            )
        torch.nn.init.xavier_uniform_(self.embedding[0].weight)

        self.convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                Conv1d(
                    in_channels= self.hp.Encoder.Size,
                    out_channels= self.hp.Encoder.Size,
                    kernel_size= self.hp.Encoder.Conv.Kernel_Size,
                    padding= (self.hp.Encoder.Conv.Kernel_Size - 1) // 2,
                    w_init_gain= 'relu'
                    ),
                torch.nn.Mish(),
                torch.nn.BatchNorm1d(
                    num_features= self.hp.Encoder.Size,
                    ),
                torch.nn.Dropout(p= self.hp.Encoder.Conv.Dropout_Rate)
                )
            for index in range(self.hp.Encoder.Conv.Stack)
            ])
        
        self.positional_encoding = Periodic_Positional_Encoding(
            period= self.hp.Encoder.Transformer.Positional_Encoding_Period,
            embedding_size= self.hp.Encoder.Size,
            dropout_rate= self.hp.Encoder.Transformer.Dropout_Rate
            )

        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Transformer.Head,
                feedforward_kernel_size= self.hp.Encoder.Transformer.FFN.Kernel_Size,
                dropout_rate= self.hp.Encoder.Transformer.Dropout_Rate,
                feedforward_dropout_rate= self.hp.Encoder.Transformer.FFN.Dropout_Rate,
                )
            for index in range(self.hp.Encoder.Transformer.Stack)
            ])

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor
        ) -> torch.Tensor:
        '''
        tokens: [Batch, Time]
        '''
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(tokens[0]).sum())).unsqueeze(1).float()

        x = self.embedding(tokens)
        for conv in self.convs:
            x = (conv(x * masks) + x) * masks
        
        x = self.positional_encoding(x) * masks
        for block in self.blocks:
            x = block(x, lengths)

        return x, masks

class Decoder(torch.nn.Sequential):
    def __init__(
        self,      
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        if self.hp.Feature_Type == 'Mel':
            feature_size = self.hp.Sound.Mel_Dim
        elif self.hp.Feature_Type == 'Spectrogram':
            feature_size = self.hp.Sound.N_FFT // 2 + 1
        
        self.positional_encoding = Periodic_Positional_Encoding(
            period= self.hp.Encoder.Transformer.Positional_Encoding_Period,
            embedding_size= self.hp.Encoder.Size,
            dropout_rate= self.hp.Encoder.Transformer.Dropout_Rate
            )

        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Transformer.Head,
                feedforward_kernel_size= self.hp.Encoder.Transformer.FFN.Kernel_Size,
                dropout_rate= self.hp.Encoder.Transformer.Dropout_Rate,
                feedforward_dropout_rate= self.hp.Encoder.Transformer.FFN.Dropout_Rate,
                )
            for index in range(self.hp.Encoder.Transformer.Stack)
            ])

        self.projection = torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= feature_size,
            kernel_size= 1,
            )

    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor
        ) -> torch.Tensor:
        '''
        encodings: [Batch, Enc_d, Enc_t]
        '''
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(encodings[0, 0]).sum())).unsqueeze(1).float()

        x = self.positional_encoding(encodings) * masks
        for block in self.blocks:
            x = block(x, lengths)

        x = self.projection(x) * masks
        
        return x, masks


class FFT_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_head: int,
        feedforward_kernel_size: int,
        dropout_rate: float= 0.1,
        feedforward_dropout_rate: float= 0.1
        ) -> None:
        super().__init__()

        self.attention = torch.nn.MultiheadAttention(
            embed_dim= channels,
            num_heads= num_head,
            dropout= dropout_rate
            )
        self.attention_norm = torch.nn.BatchNorm1d(
            num_features= channels,
            )
        
        self.ffn = FFN(
            channels= channels,
            kernel_size= feedforward_kernel_size,
            dropout_rate= feedforward_dropout_rate
            )
        self.ffn_norm = torch.nn.BatchNorm1d(
            num_features= channels,
            )
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())
        x = self.attention(
            query= x.permute(2, 0, 1),
            key= x.permute(2, 0, 1),
            value= x.permute(2, 0, 1),
            key_padding_mask= masks
            )[0].permute(1, 2, 0) + x

        masks = (~masks).unsqueeze(1).float()   # float mask
        x = self.attention_norm(x * masks)
        
        x = self.ffn(x, masks) + x
        x = self.ffn_norm(x * masks)

        return x * masks

class FFN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout_rate: float= 0.1,
        ) -> None:
        super().__init__()
        self.conv_0 = Conv1d(
            in_channels= channels,
            out_channels= channels * 4,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'relu'
            )
        self.mish = torch.nn.Mish()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv1d(
            in_channels= channels * 4,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            )
        
    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        x = self.conv_0(x * masks)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.conv_1(x * masks)

        return x * masks


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
        durations: torch.Tensor= None
        ):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        durations: [Batch, Enc_t]
        '''
        encodings = encodings.detach()
        log_duration_predictions = self.duration_predictor(encodings).squeeze(1)   # [Batch, Enc_t]
        if durations is None:
            durations = (log_duration_predictions.exp() - 1).clip(0, 50).ceil().long()
            durations[:, -1] += durations.sum(dim= 1).max() - durations.sum(dim= 1) # Align the sum of lengths

        encodings = self.length_regulator(
            encodings= encodings,
            durations= durations
            )
        
        return encodings, log_duration_predictions

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
