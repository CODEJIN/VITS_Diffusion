import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, pickle, wandb
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from librosa import griffinlim
from scipy.io import wavfile

from Modules.Modules import Model_Test, Mask_Generate, MLE_Loss

from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Noam_Scheduler import Noam_Scheduler
from Logger import Logger

from meldataset import spectral_de_normalize_torch
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.gpu_id == 0:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }

            if self.hp.Weights_and_Biases.Use:
                wandb.init(
                    project= self.hp.Weights_and_Biases.Project,
                    entity= self.hp.Weights_and_Biases.Entity,
                    name= self.hp.Weights_and_Biases.Name,
                    config= To_Non_Recursive_Dict(self.hp)
                    )
                wandb.watch(self.model)

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        feature_range_info_dict = yaml.load(open(self.hp.Spectrogram_Range_Info_Path), Loader=yaml.Loader)
        self.feature_min = min([value['Min'] for value in feature_range_info_dict.values()])
        self.feature_max = max([value['Max'] for value in feature_range_info_dict.values()])

        train_dataset = Dataset(
            token_dict= token_dict,
            feature_range_info_dict= feature_range_info_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            feature_length_min= max(self.hp.Train.Segment_Size, self.hp.Train.Train_Pattern.Feature_Length.Min),
            feature_length_max= self.hp.Train.Train_Pattern.Feature_Length.Max,
            text_length_min= self.hp.Train.Train_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Train_Pattern.Text_Length.Max,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio
            )
        eval_dataset = Dataset(
            token_dict= token_dict,
            feature_range_info_dict= feature_range_info_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            feature_length_min= max(self.hp.Train.Segment_Size, self.hp.Train.Eval_Pattern.Feature_Length.Min),
            feature_length_max= self.hp.Train.Eval_Pattern.Feature_Length.Max,
            text_length_min= self.hp.Train.Eval_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Eval_Pattern.Text_Length.Max
            )
        inference_dataset = Inference_Dataset(
            token_dict= token_dict,
            texts= self.hp.Train.Inference_in_Train.Text,
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater(
            token_dict= token_dict
            )
        inference_collater = Inference_Collater(
            token_dict= token_dict
            )

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.DistributedSampler(eval_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model = Model_Test(self.hp).to(self.device)
        self.criterion_dict = {
            'MSE': torch.nn.MSELoss(reduce= None).to(self.device),
            'MAE': torch.nn.L1Loss(reduce= None).to(self.device),
            'MLE': MLE_Loss
            }
        self.optimizer = torch.optim.NAdam(
            params= self.model.parameters(),
            lr= self.hp.Train.Learning_Rate.Initial,
            betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
            eps= self.hp.Train.ADAM.Epsilon,
            weight_decay= self.hp.Train.Weight_Decay
            )
        self.scheduler = Noam_Scheduler(
            optimizer= self.optimizer,
            warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step,
            )
        
        if self.hp.Feature_Type == 'Mel':
            self.vocoder = torch.jit.load('hifigan_ptransdifftts_exp12_310k.pts', map_location='cpu').to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        if self.gpu_id == 0:
            logging.info(self.model)

    def Train_Step(self, tokens, token_lengths, features, feature_lengths):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)
        feature_lengths = feature_lengths.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            predictions, feature_targets, log_duration_predictions, means_p, log_stds_p, durations = self.model(
                tokens= tokens,
                token_lengths= token_lengths,
                features= features,
                feature_lengths= feature_lengths
                )

            token_masks = Mask_Generate(
                lengths= token_lengths,
                max_length= tokens.size(1)
                ).to(tokens.device)

            loss_dict['Feature'] = self.criterion_dict['MAE'](
                predictions,
                feature_targets
                ).mean()
            loss_dict['Log_Duration'] = (self.criterion_dict['MSE'](
                log_duration_predictions,
                (durations.float() + 1).log()
                ) * ~token_masks).mean()
            loss_dict['MLE'] = self.criterion_dict['MLE'](
                features= features,
                feature_lengths= features,
                means_p= means_p,
                log_stds_p= log_stds_p
                )
            
        self.optimizer.zero_grad()
        self.scaler.scale(
            loss_dict['Feature'] +
            loss_dict['Log_Duration'] +
            loss_dict['MLE']
            ).backward()

        if self.hp.Train.Gradient_Norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for tokens, token_lengths, features, feature_lengths in self.dataloader_dict['Train']:
            self.Train_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                features= features,
                feature_lengths= feature_lengths
                )

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            f'Train.{key}': value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    @torch.no_grad()
    def Evaluation_Step(self, tokens, token_lengths, features, feature_lengths):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)
        feature_lengths = feature_lengths.to(self.device, non_blocking=True)

        predictions, feature_targets, log_duration_predictions, means_p, log_stds_p, durations = self.model(
            tokens= tokens,
            token_lengths= token_lengths,
            features= features,
            feature_lengths= feature_lengths
            )

        token_masks = Mask_Generate(
            lengths= token_lengths,
            max_length= tokens.size(1)
            ).to(tokens.device)

        loss_dict['Feature'] = self.criterion_dict['MAE'](
            predictions,
            feature_targets
            ).mean()
        loss_dict['Log_Duration'] = (self.criterion_dict['MSE'](
            log_duration_predictions,
            (durations.float() + 1).log()
            ) * ~token_masks).mean()
        loss_dict['MLE'] = self.criterion_dict['MLE'](
            features= features,
            feature_lengths= features,
            means_p= means_p,
            log_stds_p= log_stds_p
            )

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

        return predictions, feature_targets, log_duration_predictions, means_p, log_stds_p, durations

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model.eval()

        for step, (tokens, token_lengths, features, feature_lengths) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            predictions, feature_targets, log_duration_predictions, means_p, log_stds_p, durations = self.Evaluation_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                features= features,
                feature_lengths= feature_lengths
                )

        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            self.writer_dict['Evaluation'].add_histogram_model(self.model, 'VITS_Diffusion', self.steps, delete_keywords=[])
        
            index = np.random.randint(0, tokens.size(0))
            target_feature = (feature_targets[index].unsqueeze(0) + 1.0) / 2.0 * (self.feature_max - self.feature_min) + self.feature_min
            prediction_feature = (predictions[index].unsqueeze(0) + 1.0) / 2.0 * (self.feature_max - self.feature_min) + self.feature_min

            if self.hp.Feature_Type == 'Mel':
                target_audio = self.vocoder(target_feature.to(self.device)).squeeze(0).cpu().numpy() / 32768.0
                prediction_audio = self.vocoder(prediction_feature.to(self.device)).squeeze(0).cpu().numpy() / 32768.0
            elif self.hp.Feature_Type == 'Spectrogram':
                target_audio = griffinlim(spectral_de_normalize_torch(target_feature.squeeze(0)).cpu().numpy())
                prediction_audio = griffinlim(spectral_de_normalize_torch(prediction_feature.squeeze(0)).cpu().numpy())
            
            duration_target = durations[index, :token_lengths[index]]
            duration_target = torch.arange(duration_target.size(0)).repeat_interleave(duration_target.cpu()).numpy()
            duration_prediction = ((torch.exp(log_duration_predictions[index, :token_lengths[index]]) - 1).ceil().clip(0, 50)).long()
            duration_prediction = torch.arange(duration_prediction.size(0)).repeat_interleave(duration_prediction.cpu()).numpy()
            
            image_dict = {
                'Feature/Target': (target_feature.squeeze(0).cpu().numpy(), None, 'auto', None, None, None),
                'Feature/Prediction': (prediction_feature.squeeze(0).cpu().numpy(), None, 'auto', None, None, None),
                'Duration/Target': (duration_target[:feature_lengths[index]], None, 'auto', (0, features.size(2)), (0, tokens.size(1)), None),
                'Duration/Prediction': (duration_prediction[:feature_lengths[index]], None, 'auto', (0, features.size(2)), (0, tokens.size(1)), None),
                }
            audio_dict = {
                'Audio/Target': (target_audio, self.hp.Sound.Sample_Rate),
                'Audio/Prediction': (prediction_audio, self.hp.Sound.Sample_Rate),
                }

            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)
            self.writer_dict['Evaluation'].add_audio_dict(audio_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        f'Evaluation.{key}': value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                wandb.log(
                    data= {
                        'Evaluation.Feature.Target': wandb.Image(target_feature.squeeze(0).cpu().numpy()),
                        'Evaluation.Feature.Prediction': wandb.Image(prediction_feature.squeeze(0).cpu().numpy()),
                        'Evaluation.Duration': wandb.plot.line_series(
                            xs= np.arange(feature_lengths[index].cpu().numpy()),
                            ys= [
                                    duration_target[:feature_lengths[index]],
                                    duration_prediction[:feature_lengths[index]]
                                    ],
                            keys= ['Target', 'Prediction'],
                            title= 'Duration',
                            xname= 'Feature_t'
                            ),                    
                        'Evaluation.Audio.Target': wandb.Audio(
                            target_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Target_Audio'
                            ),
                        'Evaluation.Audio.Prediction': wandb.Audio(
                            prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Prediction_Audio'
                            ),
                        },
                    step= self.steps,
                    commit= True
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)

        self.model.train()

    @torch.no_grad()
    def Inference_Step(self, tokens, token_lengths, texts, decomposed_texts, start_index= 0, tag_step= False):
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        
        predictions, _, log_duration_predictions, *_ = self.model(
            tokens= tokens,
            token_lengths= token_lengths
            )
        predictions = (predictions + 1.0) / 2.0 * (self.feature_max - self.feature_min) + self.feature_min

        durations = (log_duration_predictions.exp() - 1).clamp(0, 50).ceil().long()
        feature_lengths = [
            int(duration[:token_length].sum())
            for duration, token_length in zip(durations, token_lengths)
            ]

        if self.hp.Feature_Type == 'Mel':
            audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))].cpu().numpy()
                for audio, length in zip(
                    self.vocoder(predictions),
                    feature_lengths
                    )
                ]
        elif self.hp.Feature_Type == 'Spectrogram':
            audios = []
            for feature, length in zip(
                predictions,
                feature_lengths
                ):
                feature = spectral_de_normalize_torch(feature).cpu().numpy()
                audio = griffinlim(feature)[:min(feature, length) * self.hp.Sound.Frame_Shift]
                audio = (audio / np.abs(audio).max() * 32767.5).astype(np.int16)
                audios.append(audio)

        files = []
        for index in range(tokens.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        durations = [
            torch.arange(duration.size(0)).repeat_interleave(duration.cpu()).numpy()
            for duration in durations
            ]

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (
            prediction,
            duration,
            token_length,
            feature_length,
            text,
            decomposed_text,
            audio,
            file
            ) in enumerate(zip(
            predictions.cpu().numpy(),
            durations,
            token_lengths.cpu().numpy(),
            feature_lengths,
            texts,
            decomposed_texts,
            audios,
            files
            )):
            title = 'Text: {}'.format(text if len(text) < 90 else text[:90] + '…')
            new_figure = plt.figure(figsize=(20, 5 * 3), dpi=100)
            ax = plt.subplot2grid((3, 1), (0, 0))
            plt.imshow(prediction[:, :feature_length], aspect='auto', origin='lower')
            plt.title('Feature prediction    {}'.format(title))
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((3, 1), (1, 0), rowspan= 2)
            plt.plot(duration[:feature_length])
            plt.title('Duration    {}'.format(title))
            plt.margins(x= 0)
            plt.yticks(
                range(len(decomposed_text) + 2),
                ['<S>'] + list(decomposed_text) + ['<E>'],
                fontsize = 10
                )
            plt.colorbar(ax= ax)
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_figure)
            
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                (audio * 32767.5).astype(np.int32)
                )
            
    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        self.model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (tokens, token_lengths, texts, decomposed_texts) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(tokens, token_lengths, texts, decomposed_texts, start_index= step * batch_size)

        self.model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])
        self.optimizer.load_state_dict(state_dict['Optimizer'])
        self.scheduler.load_state_dict(state_dict['Scheduler'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_dict = {
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps
            }
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        torch.save(state_dict, checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)

    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_path, hp_path)

        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)

        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    argParser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl'
            )
    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()