import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, pickle, base64, uuid, librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple
from librosa import griffinlim
from scipy.io import wavfile

from Modules.Modules import VITS_Diff
from Datasets import Inference_Dataset as Dataset, Inference_Collater as Collater, Text_to_Token, Token_Stack
from SSML import SSMLParser, SSMLChecker
from meldataset import mel_spectrogram, spectral_de_normalize_torch
from Arg_Parser import Recursive_Parse

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

class Inferencer:
    def __init__(
        self,
        hp_path: str,
        checkpoint_path: str,        
        batch_size= 1
        ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.model = VITS_Diff(self.hp).to(self.device)
        if self.hp.Feature_Type == 'Mel':
            self.vocoder = torch.jit.load('hifigan_ptransdifftts_exp12_500k.pts', map_location='cpu').to(self.device)
        self.ge2e_generator = torch.jit.load('GE2E_Generator.pts', map_location='cpu').to(self.device)

        if self.hp.Feature_Type == 'Spectrogram':
            self.feature_range_info_dict = yaml.load(open(self.hp.Spectrogram_Range_Info_Path), Loader=yaml.Loader)
        if self.hp.Feature_Type == 'Mel':
            self.feature_range_info_dict = yaml.load(open(self.hp.Mel_Range_Info_Path), Loader=yaml.Loader)

        if self.hp.Feature_Type == 'Spectrogram':
            self.feature_size = self.hp.Sound.N_FFT // 2 + 1
        elif self.hp.Feature_Type == 'Mel':
            self.feature_size = self.hp.Sound.Mel_Dim
        else:
            raise ValueError('Unknown feature type: {}'.format(self.hp.Feature_Type))

        self.Load_Checkpoint(checkpoint_path)
        self.batch_size = batch_size

    def Dataset_Generate(
        self,
        texts: List[str],
        speakers: List[str],
        emotions: List[str]
        ):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        emotion_info_dict = yaml.load(open(self.hp.Emotion_Info_Path), Loader=yaml.Loader)        
        ge2e_dict = pickle.load(open(self.hp.GE2E.Embedding_Dict_Path, 'rb'))
        ge2e_dict = self.GE2E_Customize(ge2e_dict)

        return torch.utils.data.DataLoader(
            dataset= Dataset(
                token_dict= token_dict,
                ge2e_dict= ge2e_dict,
                emotion_info_dict= emotion_info_dict,
                texts= texts,
                speakers= speakers,
                emotions= emotions
                ),
            shuffle= False,
            collate_fn= Collater(
                token_dict= token_dict
                ),
            batch_size= self.batch_size,
            num_workers= 0,
            pin_memory= True
            )

    def Load_Checkpoint(self, path: str):
        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])        
        self.steps = state_dict['Steps']

        self.model.eval()

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    @torch.no_grad()
    def Inference_Step(self, tokens, token_lengths, ge2es, emotions, length_scales, log_f0_scales, texts, decomposed_texts, speaker_labels, emotion_labels):
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        ge2es = ge2es.to(self.device, non_blocking=True)
        emotions = emotions.to(self.device, non_blocking=True)
        
        linear_predictions, diffusion_predictions, _, _, log_duration_predictions, _, _ = self.model(
            tokens= tokens,
            token_lengths= token_lengths,
            ge2es= ge2es,
            emotions= emotions,
            length_scales= length_scales,
            log_f0_scales= log_f0_scales
            )

        linear_predictions = linear_predictions.clamp(-1.0, 1.0)
        diffusion_predictions = diffusion_predictions.clamp(-1.0, 1.0)
        linear_prediction_list = []
        diffusion_prediction_list = []
        for linear_prediction, diffusion_prediction, speaker in zip(linear_predictions, diffusion_predictions, speaker_labels):
            feature_max = self.feature_range_info_dict[speaker]['Max']
            feature_min = self.feature_range_info_dict[speaker]['Min']
            linear_prediction_list.append((linear_prediction + 1.0) / 2.0 * (feature_max - feature_min) + feature_min)
            diffusion_prediction_list.append((diffusion_prediction + 1.0) / 2.0 * (feature_max - feature_min) + feature_min)
        linear_predictions = torch.stack(linear_prediction_list, dim= 0)
        diffusion_predictions = torch.stack(diffusion_prediction_list, dim= 0)

        durations = (log_duration_predictions.exp() - 1).clamp(0, 50).ceil().long()
        feature_lengths = [
            int(duration[:token_length].sum())
            for duration, token_length in zip(durations, token_lengths)
            ]

        if any([length < 30 for length in feature_lengths]):
            logging.warning('An inference feature length is less than 30. Inference is skipped to prevent vocoder error.')
            return

        if self.hp.Feature_Type == 'Mel':
            linear_audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))].cpu().numpy()
                for audio, length in zip(
                    self.vocoder(linear_predictions),
                    feature_lengths
                    )
                ]
            diffusion_audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))].cpu().numpy()
                for audio, length in zip(
                    self.vocoder(diffusion_predictions),
                    feature_lengths
                    )
                ]
        elif self.hp.Feature_Type == 'Spectrogram':
            linear_audios, diffusion_audios = [], []
            for linear_feature, diffusion_feature, length in zip(
                linear_predictions,
                diffusion_predictions,
                feature_lengths
                ):
                linear_feature = spectral_de_normalize_torch(linear_feature).cpu().numpy()
                linear_audio = griffinlim(linear_feature)[:min(linear_feature.size(1), length) * self.hp.Sound.Frame_Shift]
                linear_audio = (linear_audio / np.abs(linear_audio).max() * 32767.5).astype(np.int16)
                linear_audios.append(linear_audio)
                diffusion_feature = spectral_de_normalize_torch(diffusion_feature).cpu().numpy()
                diffusion_audio = griffinlim(diffusion_feature)[:min(diffusion_feature, length) * self.hp.Sound.Frame_Shift]
                diffusion_audio = (diffusion_audio / np.abs(diffusion_audio).max() * 32767.5).astype(np.int16)
                diffusion_audios.append(diffusion_audio)

        return linear_audios, diffusion_audios

    def Inference_Epoch(
        self,
        texts: List[str],
        speakers: List[str],
        emotions: List[str],
        length_scales: float= 1.0,
        log_f0_scales: float= 0.0,
        use_tqdm: bool= True
        ):
        dataloader = self.Dataset_Generate(
            texts= texts,
            speakers= speakers,
            emotions= emotions
            )
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )
        
        linear_audio_list, diffusion_audio_list = [], []
        for tokens, token_lengths, ge2es, emotions, texts, decomposed_texts, speaker_labels, emotion_labels in dataloader:
            linear_audios, diffusion_audios = self.Inference_Step(tokens, token_lengths, ge2es, emotions, length_scales, log_f0_scales, texts, decomposed_texts, speaker_labels, emotion_labels)
            linear_audio_list.extend(linear_audios)
            diffusion_audio_list.extend(diffusion_audios)

        return linear_audio_list, diffusion_audio_list


    def Wav_Reference_GE2E_Generate(self, speaker_references: List[str]):
        # Need more than 6000ms
        paths = [f'{uuid.uuid4()}.wav' for _ in range(len(speaker_references))]        
        audios = []
        for speaker_reference, path in zip(speaker_references, paths):
            open(path, 'wb').write(base64.b64decode(speaker_reference))
            audio = librosa.load(path, sr= self.hp.Sound.Sample_Rate)[0]
            audio = librosa.effects.trim(audio, top_db=60, frame_length= 512, hop_length= 256)[0]
            audio = librosa.util.normalize(audio) * 0.95
            audios.append(audio)

        for path in paths:
            os.remove(path)

        audio_lengths = [audio.shape[0] for audio in audios]
        audios = torch.from_numpy(np.array([
            np.pad(audio, [0, max(audio_lengths) - audio.shape[0]])
            for audio in audios
            ])).float()        
        features = mel_spectrogram(
            y= audios,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.Mel_Dim,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Frame_Shift,
            win_size= self.hp.Sound.Frame_Length,
            fmin= self.hp.Sound.Mel_F_Min,
            fmax= self.hp.Sound.Mel_F_Max
            )

        feature_segments = []
        for feature, audio_length in zip(features, audio_lengths):
            feature = feature[:, :audio_length // self.hp.Sound.Frame_Shift]

            frame_length = 240
            samples = 5
            if feature.shape[1] > frame_length:
                for _ in range(samples):
                    offset = np.random.randint(0, feature.shape[1] - frame_length)
                    feature_segments.append(feature[:, offset:offset + frame_length])
            else:
                pad = frame_length - feature.shape[1]
                feature_segments.extend([torch.nn.functional.pad(feature, [0, pad], mode= 'replicate')] * samples)
        feature_segments = torch.stack(feature_segments, dim= 0).to(self.device)

        return self.ge2e_generator(feature_segments, torch.LongTensor([5])).cpu()

    def Wav_Reference_Inference_Epoch(
        self,
        texts: List[str],
        speaker_references: List[str],  # base64 wav files
        emotions: List[str],
        length_scales: float= 1.0,
        log_f0_scales: float= 0.0,
        use_tqdm: bool= True
        ):
        dataloader = self.Dataset_Generate(
            texts= texts,
            speakers= ['SelectStar_Female_01'], # Dummy ge2e
            emotions= emotions
            )
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )

        ge2es = self.Wav_Reference_GE2E_Generate(speaker_references= speaker_references)
        ge2e_index = 0
        linear_audio_list, diffusion_audio_list = [], []
        for tokens, token_lengths, _, emotions, texts, decomposed_texts, speaker_labels, emotion_labels in dataloader:
            linear_audios, diffusion_audios = self.Inference_Step(tokens, token_lengths, ge2es[ge2e_index:ge2e_index + tokens.size(0)], emotions, length_scales, log_f0_scales, texts, decomposed_texts, speaker_labels, emotion_labels)
            linear_audio_list.extend(linear_audios)
            diffusion_audio_list.extend(diffusion_audios)
            ge2e_index += tokens.size(0)

        return linear_audio_list, diffusion_audio_list


    def SSML_Dataset_Generate(self, ssml_str: str):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        emotion_info_dict = yaml.load(open(self.hp.Emotion_Info_Path), Loader=yaml.Loader)        
        ge2e_dict = pickle.load(open(self.hp.GE2E.Embedding_Dict_Path, 'rb'))
        ge2e_dict = self.GE2E_Customize(ge2e_dict)

        patterns = SSMLParser().feed(' '.join(ssml_str.replace('\n', ' ').split()))

        tokens, ge2es, emotions, length_scales, log_f0_scales, energy_scales, speaker_labels = [], [], [], [], [], [], []
        for pattern in patterns:
            tokens.append(Text_to_Token(pattern['Decomposed'], token_dict))
            ge2es.append(np.stack([
                ge2e_dict[key] * weight
                for key, weight in zip(pattern['Speakers'], pattern['Weights'])
                ], axis= 0).sum(axis= 0))
            emotions.append(emotion_info_dict[pattern['Emotion']])
            length_scales.append(pattern['Duration_Scale'])
            log_f0_scales.append(pattern['Log_F0_Scale'])
            energy_scales.append(pattern['Energy_Scale'])
            speaker_labels.append(pattern['Speakers'])

        token_lengths = np.array([token.shape[0] for token in tokens])
        tokens = Token_Stack(tokens, token_dict)
        ge2es = np.array(ge2es)

        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        ge2es = torch.FloatTensor(ge2es)  # [Batch]
        emotions = torch.LongTensor(emotions)  # [Batch]

        return tokens, token_lengths, ge2es, emotions, length_scales, log_f0_scales, energy_scales, speaker_labels


    @torch.no_grad()
    def SSML_Inference(self, ssml_str: str):
        tokens, token_lengths, ge2es, emotions, length_scales, log_f0_scales, energy_scales, speaker_labels = self.SSML_Dataset_Generate(ssml_str= ssml_str)

        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        ge2es = ge2es.to(self.device, non_blocking=True)
        emotions = emotions.to(self.device, non_blocking=True)
        
        linear_predictions, diffusion_predictions, _, _, durations, _, _ = self.model.SSML(
            tokens= tokens,
            token_lengths= token_lengths,
            ge2es= ge2es,
            emotions= emotions,
            length_scales= length_scales,
            log_f0_scales= log_f0_scales,
            energy_scales= energy_scales
            )

        linear_predictions = linear_predictions.clamp(-1.0, 1.0)
        diffusion_predictions = diffusion_predictions.clamp(-1.0, 1.0)
        linear_prediction_list = []
        diffusion_prediction_list = []
        for linear_prediction, diffusion_prediction, speaker in zip(linear_predictions, diffusion_predictions, speaker_labels):
            feature_max = max([self.feature_range_info_dict[x]['Max'] for x in speaker])
            feature_min = min([self.feature_range_info_dict[x]['Min'] for x in speaker])
            linear_prediction_list.append((linear_prediction + 1.0) / 2.0 * (feature_max - feature_min) + feature_min)
            diffusion_prediction_list.append((diffusion_prediction + 1.0) / 2.0 * (feature_max - feature_min) + feature_min)
        linear_predictions = torch.stack(linear_prediction_list, dim= 0)
        diffusion_predictions = torch.stack(diffusion_prediction_list, dim= 0)

        feature_lengths = [
            int(duration[:token_length].sum())
            for duration, token_length in zip(durations, token_lengths)
            ]

        if self.hp.Feature_Type == 'Mel':
            linear_audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))]
                for audio, length in zip(
                    self.vocoder(linear_predictions),
                    feature_lengths
                    )
                ]
            diffusion_audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))]
                for audio, length in zip(
                    self.vocoder(diffusion_predictions),
                    feature_lengths
                    )
                ]
        elif self.hp.Feature_Type == 'Spectrogram':
            linear_audios, diffusion_audios = [], []
            for linear_feature, diffusion_feature, length in zip(
                linear_predictions,
                diffusion_predictions,
                feature_lengths
                ):
                linear_feature = spectral_de_normalize_torch(linear_feature)
                linear_audio = griffinlim(linear_feature)[:min(linear_feature.size(1), length) * self.hp.Sound.Frame_Shift]
                linear_audio = (linear_audio / linear_audio.abs().max() * 32767.5).short()
                linear_audios.append(linear_audio)
                diffusion_feature = spectral_de_normalize_torch(diffusion_feature)
                diffusion_audio = griffinlim(diffusion_feature)[:min(diffusion_feature, length) * self.hp.Sound.Frame_Shift]
                diffusion_audio = (diffusion_audio / diffusion_audio.abs().max() * 32767.5).short()
                diffusion_audios.append(diffusion_audio)

        linear_audio = torch.cat(linear_audios, dim= 0).cpu().numpy()
        diffusion_audio = torch.cat(diffusion_audios, dim= 0).cpu().numpy()

        return linear_audio, diffusion_audio

    def GE2E_Customize(self, ge2e_dict):
        ge2e_dict['SGHVC_Yura'] = \
            ge2e_dict['SGHVC_Yura'] * 0.90 + \
            ge2e_dict['SelectStar_Female_02'] * 0.10
        ge2e_dict['SGHVC_Yura_C'] = \
            ge2e_dict['SGHVC_Yura'] * 0.10 + \
            ge2e_dict['SGHVC_Yura_P2'] * 0.75 + \
            ge2e_dict['SelectStar_Female_02'] * 0.15
        ge2e_dict['Tamarinne_Normal'] = \
            ge2e_dict['Tamarinne_Normal'] * 0.75 + \
            ge2e_dict['SelectStar_SYW'] * 0.25
        ge2e_dict['Tamarinne_Normal_P2'] = \
            ge2e_dict['Tamarinne_Normal_P2'] * 0.75 + \
            ge2e_dict['SelectStar_SYW'] * 0.25
        ge2e_dict['Tamarinne_Songstress'] = \
            ge2e_dict['Tamarinne_Songstress'] * 0.75 + \
            ge2e_dict['SelectStar_SYW'] * 0.25
        ge2e_dict['Tamarinne_Songstress_P2'] = \
            ge2e_dict['Tamarinne_Songstress_P2'] * 0.75 + \
            ge2e_dict['SelectStar_SYW'] * 0.25
            
        self.feature_range_info_dict['SGHVC_Yura_C'] = self.feature_range_info_dict['SGHVC_Yura']

        return ge2e_dict