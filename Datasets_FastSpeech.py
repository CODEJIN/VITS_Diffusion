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

def Duration_Stack(durations):
    max_duration_length = max([duration.shape[0] for duration in durations])
    max_duration_sum = max([duration.sum() for duration in durations])
    durations = np.stack(
        [np.pad(duration, [1, max_duration_length - duration.shape[0] + 1]) for duration in durations],
        axis= 0
        )   # <S>,<E>
    durations[:, -1] = durations[:, -1] + max_duration_sum - durations.sum(axis=1)

    return durations

def Feature_Stack(features, max_length: int= None):
    max_feature_length = max_length or max([feature.shape[0] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, max_feature_length - feature.shape[0]], [0, 0]], constant_values= -1.0) for feature in features],
        axis= 0
        )
    return features



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
        duration_dict: Dict[str, np.array],
        pattern_path: str,
        metadata_file: str,
        feature_type: str,
        feature_length_min: int,
        feature_length_max: int,
        text_length_min: int,
        text_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0
        ):
        super().__init__()
        self.token_dict = token_dict
        self.feature_min = min([value['Min'] for value in feature_range_info_dict.values()])
        self.feature_max = max([value['Max'] for value in feature_range_info_dict.values()])
        self.duration_dict = duration_dict
        self.feature_type = feature_type
        self.pattern_path = pattern_path
        
        if feature_type == 'Mel':
            feature_length_dict = 'Mel_Length_Dict'
        elif feature_type == 'Spectrogram':
            feature_length_dict = 'Spectrogram_Length_Dict'

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
                metadata_dict[feature_length_dict][x] >= feature_length_min,
                metadata_dict[feature_length_dict][x] <= feature_length_max,
                metadata_dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_dict['Text_Length_Dict'][x] <= text_length_max,
                x in duration_dict.keys()
                ])
            ] * accumulated_dataset_epoch

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        pattern_dict = pickle.load(open(path, 'rb'))
        
        token = Text_to_Token(pattern_dict['Decomposed'], self.token_dict)
        feature = pattern_dict[self.feature_type]
        feature = (feature - self.feature_min) / (self.feature_max - self.feature_min) * 2.0 - 1.0

        duration = self.duration_dict[self.patterns[idx]]
        if duration.sum() < feature.shape[0]:
            duration[-1] += feature.shape[0] - duration.sum()
        elif duration.sum() > feature.shape[0]:
            print(path, duration.sum(), feature.shape[0])
            assert False

        return token, feature, duration

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        texts: List[str]
        ):
        super().__init__()
        self.token_dict = token_dict

        self.patterns = []
        for index, text in enumerate(texts):
            text = Text_Filtering(text)

            if text is None or text == '':
                logging.warning('The text of index {} is incorrect. This index is ignoired.'.format(index))
                continue

            self.patterns.append(text)

    def __getitem__(self, idx):
        text = self.patterns[idx]        
        decomposed_text = Decompose(text)
        
        return Text_to_Token(decomposed_text, self.token_dict), text, decomposed_text

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int]
        ):
        self.token_dict = token_dict

    def __call__(self, batch):
        tokens, features, durations = zip(*batch)
        token_lengths = np.array([token.shape[0] for token in tokens])
        feature_lengths = np.array([feature.shape[0] for feature in features])

        tokens = Token_Stack(tokens, self.token_dict)
        features = Feature_Stack(features)
        durations = Duration_Stack(durations).astype(np.int32)

        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Feature_d, Featpure_t]
        feature_lengths = torch.LongTensor(feature_lengths)   # [Batch]
        durations = torch.LongTensor(durations)   # [Batch, Token_t]

        return tokens, token_lengths, features, feature_lengths, durations

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, texts, decomposed_texts = zip(*batch)
        
        token_lengths = np.array([token.shape[0] for token in tokens])
        
        tokens = Token_Stack(tokens, self.token_dict)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        
        return tokens, token_lengths, texts, decomposed_texts