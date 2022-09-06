from argparse import Namespace
import torch
import numpy as np
import pickle, os, logging
from typing import Dict, List

from Pattern_Generator import Text_Filtering, Decompose

def Text_to_Token(text, token_dict):
    return np.array([
        token_dict[letter]
        for letter in ['<S>'] + list(text) + ['<E>']
        ], dtype= np.int32)

def Token_Stack(tokens, token_dict, max_length: int= None):
    max_token_length = max_length or max([token.shape[0] for token in tokens])
    tokens = np.stack(
        [np.pad(token, [0, max_token_length - token.shape[0]], constant_values= token_dict['<E>']) for token in tokens],
        axis= 0
        )
    return tokens

def Feature_Stack(features, max_length: int= None):
    max_feature_length = max_length or max([feature.shape[0] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, max_feature_length - feature.shape[0]], [0, 0]], constant_values= -1.0) for feature in features],
        axis= 0
        )
    return features

def Log_F0_Stack(log_f0s):
    max_log_f0_length = max([log_f0.shape[0] for log_f0 in log_f0s])
    log_f0s = np.stack(
        [np.pad(log_f0, [0, max_log_f0_length - log_f0.shape[0]], constant_values= -5.0) for log_f0 in log_f0s],
        axis= 0
        )
    return log_f0s

def Energy_Stack(energies):
    max_energy_length = max([energy.shape[0] for energy in energies])
    energies = np.stack(
        [np.pad(energy, [0, max_energy_length - energy.shape[0]], constant_values= -1.5) for energy in energies],
        axis= 0
        )
    return energies

def Audio_Stack(audios):
    max_audio_length = max([audio.shape[0] for audio in audios])
    audios = np.stack(
        [np.pad(audio, [0, max_audio_length - audio.shape[0]]) for audio in audios],
        axis= 0
        )
    return audios

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        feature_range_info_dict: Dict[str, Dict[str, float]],
        log_f0_info_dict: Dict[str, Dict[str, float]],
        energy_info_dict: Dict[str, Dict[str, float]],
        ge2e_dict: Dict[str, int],
        emotion_info_dict: Dict[str, int],
        pattern_path: str,
        metadata_file: str,
        feature_length_min: int,
        feature_length_max: int,
        text_length_min: int,
        text_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0
        ):
        super().__init__()
        self.token_dict = token_dict
        self.feature_range_info_dict = feature_range_info_dict
        self.log_f0_info_dict = log_f0_info_dict
        self.energy_info_dict = energy_info_dict
        self.ge2e_dict = ge2e_dict
        self.emotion_info_dict = emotion_info_dict
        self.pattern_path = pattern_path
        
        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))
        
        self.patterns = []
        max_pattern_by_speaker = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Speaker_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Speaker_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_speaker)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        self.patterns = [
            x for x in self.patterns
            if all([
                metadata_dict['Spectrogram_Length_Dict'][x] >= feature_length_min,
                metadata_dict['Spectrogram_Length_Dict'][x] <= feature_length_max,
                metadata_dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_dict['Text_Length_Dict'][x] <= text_length_max
                ])
            ] * accumulated_dataset_epoch

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        pattern_dict = pickle.load(open(path, 'rb'))
        speaker = pattern_dict['Speaker']
        emotion = pattern_dict['Emotion']
        
        token = Text_to_Token(pattern_dict['Decomposed'], self.token_dict)
        feature = pattern_dict['Spectrogram']
        # feature_min = self.feature_range_info_dict[speaker]['Min']
        # feature_max = self.feature_range_info_dict[speaker]['Max']
        # feature = (feature - feature_min) / (feature_max - feature_min) * 2.0 - 1.0

        log_f0 = (pattern_dict['Log_F0'] - self.log_f0_info_dict[speaker]['Mean']) / self.log_f0_info_dict[speaker]['Std']
        log_f0 = np.clip(log_f0, -5.0, np.inf)
        energy = (pattern_dict['Energy'] - self.energy_info_dict[speaker]['Mean']) / self.energy_info_dict[speaker]['Std']
        energy = np.clip(energy, -1.5, np.inf)

        return token, self.ge2e_dict[speaker], self.emotion_info_dict[emotion], feature, log_f0, energy, pattern_dict['Audio']

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        ge2e_dict: Dict[str, int],
        emotion_info_dict: Dict[str, int],
        texts: List[str],
        speakers: List[str],
        emotions: List[str]
        ):
        super().__init__()
        self.token_dict = token_dict
        self.ge2e_dict = ge2e_dict
        self.emotion_info_dict = emotion_info_dict

        self.patterns = []
        for index, (text, speaker, emotion) in enumerate(zip(texts, speakers, emotions)):
            text = Text_Filtering(text)

            if text is None or text == '':
                logging.warning('The text of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            if not speaker in self.ge2e_dict.keys():
                logging.warning('The speaker of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            if not emotion in self.emotion_info_dict.keys():
                logging.warning('The emotion of index {} is incorrect. This index is ignoired.'.format(index))
                continue

            self.patterns.append((text, speaker, emotion))

    def __getitem__(self, idx):
        text, speaker, emotion = self.patterns[idx]        
        decomposed_text = Decompose(text)
        
        return Text_to_Token(decomposed_text, self.token_dict), self.ge2e_dict[speaker], self.emotion_info_dict[emotion], text, decomposed_text, speaker, emotion

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int]
        ):
        self.token_dict = token_dict

    def __call__(self, batch):
        tokens, ge2es, emotions, features, log_f0s, energies, audios  = zip(*batch)
        token_lengths = np.array([token.shape[0] for token in tokens])
        feature_lengths = np.array([feature.shape[0] for feature in features])
        audio_lengths = np.array([audio.shape[0] for audio in audios])

        tokens = Token_Stack(tokens, self.token_dict)
        ge2es = np.array(ge2es)

        features = Feature_Stack(features)
        log_f0s = Log_F0_Stack(log_f0s)
        energies = Energy_Stack(energies)
        audios = Audio_Stack(audios)

        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Feature_d, Featpure_t]
        ge2es = torch.FloatTensor(ge2es)   # [Batch, GE2E_dim]
        emotions = torch.LongTensor(emotions)  # [Batch]
        feature_lengths = torch.LongTensor(feature_lengths)   # [Batch]
        log_f0s = torch.FloatTensor(log_f0s)    # [Batch, Feature_t]
        energies = torch.FloatTensor(energies)  # [Batch, Feature_t]
        audios = torch.FloatTensor(audios)  # [Batch, Audio_t]
        audio_lengths = torch.LongTensor(audio_lengths)   # [Batch]

        return tokens, token_lengths, ge2es, emotions, features, feature_lengths, log_f0s, energies, audios, audio_lengths

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, ge2es, emotions, texts, decomposed_texts, speaker_lables, emotion_labels = zip(*batch)
        
        token_lengths = np.array([token.shape[0] for token in tokens])
        
        tokens = Token_Stack(tokens, self.token_dict)
        ge2es = np.array(ge2es)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        ge2es = torch.FloatTensor(ge2es)  # [Batch]
        emotions = torch.LongTensor(emotions)  # [Batch]
        
        return tokens, token_lengths, ge2es, emotions, texts, decomposed_texts, speaker_lables, emotion_labels