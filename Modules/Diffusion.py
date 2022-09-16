import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm

from .Layer import Conv1d, ConvTranspose2d, Lambda

class Diffusion(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.denoiser = Denoiser(
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
        conditions: torch.Tensor,
        audios: torch.Tensor= None
        ):
        '''
        conditions: [Batch, Enc_d, Feature_t]
        audios: [Batch, Audio_t]
        Audio_t = Feature_t * Hop_size
        '''
        if not audios is None:    # train
            diffusion_steps = torch.randint(
                low= 0,
                high= self.timesteps,
                size= (conditions.size(0),),
                dtype= torch.long,
                device= conditions.device
                )    # random single step
            
            noises, epsilons = self.Get_Noise_Epsilon_for_Train(
                audios= audios,
                conditions= conditions,
                diffusion_steps= diffusion_steps,
                )
            return None, noises, epsilons
        else:   # inference
            audios = self.Sampling(
                conditions= conditions,
                )
            return audios, None, None

    def Sampling(
        self,
        conditions: torch.Tensor,        
        ):
        audios = torch.randn(
            size= (conditions.size(0), conditions.size(2) * self.hp.Sound.Frame_Shift),
            device= conditions.device
            )
        for diffusion_step in tqdm(
            reversed(range(self.timesteps)),
            desc= '[Diffusion]',
            total= self.timesteps
            ):
            audios = self.P_Sampling(
                audios= audios,
                conditions= conditions,
                diffusion_steps= torch.full(
                    size= (conditions.size(0), ),
                    fill_value= diffusion_step,
                    dtype= torch.long,
                    device= conditions.device
                    ),
                )
        
        return audios
        
    def P_Sampling(
        self,
        conditions: torch.Tensor,
        diffusion_steps: torch.Tensor,
        audios: torch.Tensor,        
        ):
        posterior_means, posterior_log_variances = self.Get_Posterior(
            audios= audios,
            diffusion_steps= diffusion_steps,
            conditions= conditions,
            )

        noises = torch.randn_like(audios)   # [Batch, Audio_t]
        masks = (diffusion_steps > 0).float().unsqueeze(1)  #[Batch, 1]
        
        return posterior_means + masks * (0.5 * posterior_log_variances).exp() * noises

    def Get_Posterior(
        self,
        audios: torch.Tensor,
        conditions: torch.Tensor,
        diffusion_steps: torch.Tensor
        ):
        noised_predictions = self.denoiser(
            audios= audios,
            conditions= conditions,
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
        conditions: torch.Tensor,
        diffusion_steps: torch.Tensor,
        ):
        noises = torch.randn_like(audios)

        noised_audios = \
            audios * self.sqrt_alphas_cumprod[diffusion_steps][:, None] + \
            noises * self.sqrt_one_minus_alphas_cumprod[diffusion_steps][:, None]

        epsilons = self.denoiser(
            audios= noised_audios,
            conditions= conditions,
            diffusion_steps= diffusion_steps
            )
        
        return noises, epsilons


class Denoiser(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters
        assert math.prod(self.hp.Diffusion.Stride) == self.hp.Sound.Frame_Shift

        self.prenet = torch.nn.Sequential(
            Lambda(lambda x: x.unsqueeze(1)),
            torch.nn.utils.weight_norm(Conv1d(
                in_channels= 1,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                w_init_gain= 'relu'
                )),
            torch.nn.ReLU()
            )    

        self.diffusion_embedding = torch.nn.Sequential(
            Diffusion_Embedding(
                channels= self.hp.Diffusion.Size
                ),
            Lambda(lambda x: x.unsqueeze(2))
            )
        self.embedding_ffn = torch.nn.Sequential(
            Conv1d(
                in_channels= self.hp.Diffusion.Size,
                out_channels= self.hp.Diffusion.Size * 4,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.Mish(),
            Conv1d(
                in_channels= self.hp.Diffusion.Size * 4,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                w_init_gain= 'linear'
                ),
            )

        self.blocks = torch.nn.ModuleList([
            Residual_Block(
                channels= self.hp.Diffusion.Size,
                condition_channels= self.hp.Encoder.Size,
                kernel_size= self.hp.Diffusion.Kernel_Size,
                dilation= 2 ** (index % self.hp.Diffusion.Dilation_Cycle),
                strides= self.hp.Diffusion.Stride,
                leaky_relu_slope= self.hp.Diffusion.Leaky_ReLU_Slope,
                )
            for index in range(self.hp.Diffusion.Stack)
            ])

        self.projection = torch.nn.Sequential(
            torch.nn.utils.weight_norm(Conv1d(
                in_channels= self.hp.Diffusion.Size,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                w_init_gain= 'relu'
                )),
            torch.nn.ReLU(),
            Conv1d(
                in_channels= self.hp.Diffusion.Size,
                out_channels= 1,
                kernel_size= 1
                ),
            Lambda(lambda x: x.squeeze(1))
            )
        torch.nn.init.zeros_(self.projection[-2].weight)    # This is key factor....
        torch.nn.init.zeros_(self.projection[-2].bias)    # This is key factor....

    def forward(
        self,
        audios: torch.Tensor,
        conditions: torch.Tensor,
        diffusion_steps: torch.Tensor
        ):
        '''
        x: [Batch, Audio_t]
        conditions: Feautre, [Batch, Enc_d, Feature_t]
        diffusion_steps: [Batch]
        '''
        audios = self.prenet(audios)  # [Batch, Diffusion_d, Audio_t]
        diffusion_steps = self.diffusion_embedding(diffusion_steps)   # [Batch, Diffusion_d, 1]
        diffusion_steps = self.embedding_ffn(diffusion_steps) # [Batch, Diffusion_d, 1]
        
        skips_list = []
        for block in self.blocks:
            audios, skips = block(
                x= audios,
                conditions= conditions,
                diffusion_steps= diffusion_steps
                )   # [Batch, Diffusion, Audio_t], [Batch, Diffusion, Audio_t]
            skips_list.append(skips)

        audios = torch.stack(skips_list, dim= 0).sum(dim= 0) / math.sqrt(self.hp.Diffusion.Stack)
        audios = self.projection(audios)  # [Batch, Audio_t]

        return audios

class Diffusion_Embedding(torch.nn.Module):
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
        channels: int,
        condition_channels: int,
        kernel_size: int,
        dilation: int,
        strides: List[int],  # upsample rate
        leaky_relu_slope: float= 0.4
        ):
        super().__init__()
        self.diffusion_embedding = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= 1
            )

        self.dilated_conv = torch.nn.utils.weight_norm(Conv1d(
            in_channels= channels,
            out_channels= channels * 2,
            kernel_size= kernel_size,
            dilation= dilation,
            padding= dilation * (kernel_size - 1) // 2,
            w_init_gain= 'gate'
            ))

        self.condition = torch.nn.Sequential()
        self.condition.add_module('Unsqueeze', Lambda(lambda x: x.unsqueeze(1)))
        for index, stride in enumerate(strides):
            self.condition.add_module(f'Upsample_{index}', torch.nn.utils.weight_norm(ConvTranspose2d(
                in_channels= 1,
                out_channels= 1,
                kernel_size= (kernel_size, stride * 2),
                stride= (1, stride),
                padding= ((kernel_size - 1) // 2, stride // 2)
                )))
            self.condition.add_module(f'LeakyReLU_{index}', torch.nn.LeakyReLU(
                negative_slope= leaky_relu_slope
                ))
        self.condition.add_module('Squeeze', Lambda(lambda x: x.squeeze(1)))
        self.condition.add_module('Conv', torch.nn.utils.weight_norm(Conv1d(
            in_channels= condition_channels,
            out_channels= channels * 2,
            kernel_size= 1,
            w_init_gain= 'gate'
            )))

        self.projection = Conv1d(
            in_channels= channels,
            out_channels= channels * 2,
            kernel_size= 1,
            w_init_gain= 'linear'
            )

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor,
        diffusion_steps: torch.Tensor
        ):
        '''
        x: [Batch, Diffusion_d, Audio_t]
        conditions: Feautre, [Batch, Enc_d, Feature_t]
        diffusion_steps: [Batch, Diffusion_d, 1]
        '''
        residuals = x

        diffusion_steps = self.diffusion_embedding(diffusion_steps) # [Batch, Diffusion_d, 1]
        x = self.dilated_conv(x + diffusion_steps) + self.condition(conditions)   # [Batch, Diffusion_d * 2, Audio_t]
        
        x_a, x_b = x.chunk(chunks= 2, dim= 1)
        x = x_a.tanh() * x_b.sigmoid()

        x = self.projection(x)
        x, skips = x.chunk(chunks= 2, dim= 1)

        return (x + residuals) / math.sqrt(2.0), skips
