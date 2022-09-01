# MIT License
#
# Copyright (c) 2021 Keon Lee
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# https://github.com/keonlee9420/DiffSinger
#
# The base is from Keon Lee

import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from .Layer import Conv1d, ConvTranspose1d, Lambda

# from Source_Diffusions import Denoiser

class Difussion(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.denoiser = Denoiser(
            hyper_parameters= self.hp
            )
        self.upsampler = Upsampler(
            hyper_parameters= self.hp
            )

        self.timesteps = self.hp.Diffusion.Max_Step
        betas = torch.linspace(1e-4, 0.06, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis= 0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())  # [Diffusion_t]
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())    # [Diffusion_t]
        self.register_buffer('sqrt_recip_alphas_cumprod', (1.0 / alphas_cumprod).sqrt())    # [Diffusion_t]
        self.register_buffer('sqrt_recipm1_alphas_cumprod', (1.0 / alphas_cumprod - 1.0).sqrt())    # [Diffusion_t]

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance', torch.maximum(posterior_variance, torch.tensor([1e-20])).log())  # [Diffusion_t]
        self.register_buffer('posterior_mean_coef1', betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod))   # [Diffusion_t]
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod))  # [Diffusion_t]

    def forward(
        self,
        encodings: torch.Tensor,
        audios: torch.Tensor= None
        ):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        audios: [Batch, Audio_t]
        Audio_t = Enc_t * Hop size
        '''
        encodings = self.upsampler(encodings)   # [Batch, Enc_d, Audio_t]

        if not audios is None:    # train
            diffusion_steps = torch.randint(
                low= 0,
                high= self.timesteps,
                size= (encodings.size(0),),
                dtype= torch.long,
                device= encodings.device
                )    # random single step
            
            noises, epsilons = self.Get_Noise_Epsilon_for_Train(
                audios= audios,
                encodings= encodings,
                diffusion_steps= diffusion_steps,
                )
            return None, noises, epsilons
        else:   # inference
            audios = self.Sampling(
                encodings= encodings,
                )
            return audios, None, None

    def Sampling(
        self,
        encodings: torch.Tensor,        
        ):
        audios = torch.randn(
            size= (encodings.size(0), encodings.size(2)),
            device= encodings.device
            )
        for diffusion_step in reversed(range(self.timesteps)):
            audios = self.P_Sampling(
                audios= audios,
                encodings= encodings,
                diffusion_steps= torch.full(
                    size= (encodings.size(0), ),
                    fill_value= diffusion_step,
                    dtype= torch.long,
                    device= encodings.device
                    ),
                )
        
        return audios
        
    def P_Sampling(
        self,
        encodings: torch.Tensor,
        diffusion_steps: torch.Tensor,
        audios: torch.Tensor,        
        ):
        posterior_means, posterior_log_variances = self.Get_Posterior(
            audios= audios,
            diffusion_steps= diffusion_steps,
            encodings= encodings,
            )

        noises = torch.randn_like(audios) # [Batch, Audio_t]
        masks = (diffusion_steps > 0).float().unsqueeze(1) #[Batch, 1]
        
        return posterior_means + masks * (0.5 * posterior_log_variances).exp() * noises

    def Get_Posterior(
        self,
        audios: torch.Tensor,
        encodings: torch.Tensor,
        diffusion_steps: torch.Tensor
        ):
        noised_predictions = self.denoiser(
            audios= audios,
            encodings= encodings,
            diffusion_steps= diffusion_steps
            )

        epsilons = \
            audios * self.sqrt_recip_alphas_cumprod[diffusion_steps][:, None] - \
            noised_predictions * self.sqrt_recipm1_alphas_cumprod[diffusion_steps][:, None] # [Batch, Audio_t]
        epsilons.clamp_(-1.0, 1.0)  # clipped
        
        posterior_means = \
            epsilons * self.posterior_mean_coef1[diffusion_steps][:, None] + \
            audios * self.posterior_mean_coef2[diffusion_steps][:, None]    # [Batch, Audio_t]
        posterior_log_variances = \
            self.posterior_log_variance[diffusion_steps][:, None]   # [Batch, Audio_t]
        
        return posterior_means, posterior_log_variances

    def Get_Noise_Epsilon_for_Train(
        self,
        audios: torch.Tensor,
        encodings: torch.Tensor,
        diffusion_steps: torch.Tensor,
        ):
        noises = torch.randn_like(audios)

        noised_audios = \
            audios * self.sqrt_alphas_cumprod[diffusion_steps][:, None] + \
            noises * self.sqrt_one_minus_alphas_cumprod[diffusion_steps][:, None]

        epsilons = self.denoiser(
            audios= noised_audios,
            encodings= encodings,
            diffusion_steps= diffusion_steps
            )
        
        return noises, epsilons


class Denoiser(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = torch.nn.Sequential(
            Lambda(lambda x: x.unsqueeze(1)),
            Conv1d(
                in_channels= 1,
                out_channels= self.hp.Diffusion.Denoiser.Channels,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.Mish()
            )

        self.diffusion_embedding = DiffusionEmbedding(
            channels= self.hp.Diffusion.Denoiser.Channels
            )
        self.diffusion_ffn = torch.nn.Sequential(
            Conv1d(
                in_channels= self.hp.Diffusion.Denoiser.Channels,
                out_channels= self.hp.Diffusion.Denoiser.Channels * 4,
                kernel_size= 1,
                w_init_gain= 'linear'
                ),
            torch.nn.Mish(),
            Conv1d(
                in_channels= self.hp.Diffusion.Denoiser.Channels * 4,
                out_channels= self.hp.Diffusion.Denoiser.Channels,
                kernel_size= 1,
                w_init_gain= 'linear'
                ),
            )

        self.residual_blocks = torch.nn.ModuleList([
            Residual_Block(
                in_channels= self.hp.Diffusion.Denoiser.Channels,
                kernel_size= self.hp.Diffusion.Denoiser.Kernel_Size,
                encoding_channels= self.hp.Encoder.Size,
                )
            for _ in range(self.hp.Diffusion.Denoiser.Stack)
            ])
        
        self.projection =  torch.nn.Sequential(
            Conv1d(
                in_channels= self.hp.Diffusion.Denoiser.Channels,
                out_channels= self.hp.Diffusion.Denoiser.Channels,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.Mish(),
            Conv1d(
                in_channels= self.hp.Diffusion.Denoiser.Channels,
                out_channels= 1,
                kernel_size= 1
                ),
            Lambda(lambda x: x.squeeze(1))
            )
        torch.nn.init.zeros_(self.projection[-2].weight)    # This is key factor....
            
    def forward(
        self,
        audios: torch.Tensor,
        encodings: torch.Tensor,
        diffusion_steps: torch.Tensor
        ):
        '''
        audios: [Batch, Audio_t]
        encodings: [Batch, Enc_d, Feature_t]
        diffusion_steps: [Batch]
        '''
        x = self.prenet(audios)
        
        diffusions = self.diffusion_embedding(diffusion_steps).unsqueeze(2) # [Batch, Res_d, 1]
        diffusions = self.diffusion_ffn(diffusions) # [Batch, Res_d, 1]
        
        skips_list = []
        for residual_block in self.residual_blocks:
            x, skips = residual_block(
                x= x,
                conditions= encodings,
                diffusions= diffusions
                )
            skips_list.append(skips)

        x = torch.stack(skips_list, dim= 0).sum(dim= 0) / math.sqrt(self.hp.Diffusion.Denoiser.Stack)
        x = self.projection(x)

        return x

class DiffusionEmbedding(torch.nn.Module):
    def __init__(
        self,
        channels: int
        ):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor):
        half_channels = self.channels // 2  # sine and cosine
        embeddings = math.log(10000.0) / (half_channels - 1)
        embeddings = torch.exp(torch.arange(half_channels, device= x.device) * -embeddings)
        embeddings = x.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim= -1)

        return embeddings

class Residual_Block(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        encoding_channels: int
        ):
        super().__init__()
        self.in_channels = in_channels
        
        self.condition = Conv1d(
            in_channels= encoding_channels,
            out_channels= in_channels * 2,
            kernel_size= 1
            )
        self.diffusion = Conv1d(
            in_channels= in_channels,
            out_channels= in_channels,
            kernel_size= 1
            )

        self.conv = Conv1d(
            in_channels= in_channels,
            out_channels= in_channels * 2,
            kernel_size= kernel_size,
            padding= kernel_size // 2
            )

        self.projection = Conv1d(
            in_channels= in_channels,
            out_channels= in_channels * 2,
            kernel_size= 1
            )

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor,
        diffusions: torch.Tensor
        ):
        residuals = x

        conditions = self.condition(conditions)
        diffusions = self.diffusion(diffusions)

        x = self.conv(x + diffusions) + conditions
        x_a, x_b = x.chunk(chunks= 2, dim= 1)
        x = x_a.sigmoid() * x_b.tanh()

        x = self.projection(x)
        x, skips = x.chunk(chunks= 2, dim= 1)

        return (x + residuals) / math.sqrt(2.0), skips


class Upsampler(torch.nn.Sequential):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        assert math.prod(self.hp.Diffusion.Upsampler.Stride) == self.hp.Sound.Frame_Shift

        for index, stride in enumerate(self.hp.Diffusion.Upsampler.Stride):
            self.add_module(f'ConvTranspose_{index}', ConvTranspose1d(                
                in_channels= self.hp.Encoder.Size,
                out_channels= self.hp.Encoder.Size,
                kernel_size= self.hp.Diffusion.Upsampler.Kernel_Size,
                stride= stride,
                padding= \
                    (self.hp.Diffusion.Upsampler.Kernel_Size - stride) // 2 + \
                    (self.hp.Diffusion.Upsampler.Kernel_Size - stride) % 2,
                w_init_gain= 'leaky_relu'
                ))
            if index < len(self.hp.Diffusion.Upsampler.Stride) - 1:
                self.add_module(f'Mish_{index}', torch.nn.Mish())

    def forward(self, encodings: torch.Tensor):
        return super().forward(encodings)
