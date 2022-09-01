import torch
import numpy as np
import yaml, os, pickle, librosa, re, argparse, math
from concurrent.futures import ThreadPoolExecutor as PE
from random import shuffle
from tqdm import tqdm
import hgtk
from pysptk.sptk import rapt

from meldataset import mel_spectrogram, spectrogram, spec_energy
from Arg_Parser import Recursive_Parse

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
regex_Checker = re.compile('[가-힣A-Z,.?!\'\-\s]+')
top_db_dict = {'KSS': 35, 'Emotion': 30, 'AIHub': 30, 'Seoul': 20, 'YUA': 40, 'JPS': 40, 'LostArk': 40, 'VCTK': 15, 'Libri': 23}

def Text_Filtering(text):
    remove_Letter_List = ['(', ')', '\"', '[', ']', ':', ';']
    replace_List = [('  ', ' '), (' ,', ','), ('\' ', '\''), ('“', ''), ('”', ''), ('’', '\'')]

    text = text.upper().strip()
    for filter in remove_Letter_List:
        text= text.replace(filter, '')
    for filter, replace_STR in replace_List:
        text= text.replace(filter, replace_STR)

    text= text.strip()
    
    if len(regex_Checker.findall(text)) != 1:
        return None
    elif text.startswith('\''):
        return None
    else:
        return regex_Checker.findall(text)[0]

def Decompose(text):
    decomposed = []
    for letter in text:
        if not hgtk.checker.is_hangul(letter):
            decomposed.append(letter)
            continue

        onset, nucleus, coda = hgtk.letter.decompose(letter)
        coda += '_'
        decomposed.extend([onset, nucleus, coda])

    return decomposed

def Pattern_Generate(
    path,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int,
    f0_min: int,
    f0_max: int,
    center: bool= False,
    top_db= 60
    ):

    audio, _ = librosa.load(path, sr= sample_rate)
    audio = librosa.effects.trim(audio, top_db=top_db, frame_length= 512, hop_length= 256)[0]
    audio = librosa.util.normalize(audio) * 0.95
    audio = audio[:audio.shape[0] - (audio.shape[0] % hop_size)]
    spect = spectrogram(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= n_fft,
        hop_size= hop_size,
        win_size= win_size,
        center= center
        ).squeeze(0).T.numpy()
    mel = mel_spectrogram(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= n_fft,
        num_mels= num_mels,
        sampling_rate= sample_rate,
        hop_size= hop_size,
        win_size= win_size,
        fmin= fmin,
        fmax= fmax,
        center= center
        ).squeeze(0).T.numpy()

    log_f0 = rapt(
        x= audio * 32768,
        fs= sample_rate,
        hopsize= hop_size,
        min= f0_min,
        max= f0_max,
        otype= 2    # log
        )

    energy = spec_energy(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= n_fft,
        hop_size= hop_size,
        win_size= win_size,
        center= center
        ).squeeze(0).numpy()

    if log_f0.shape[0] != mel.shape[0]:
        print(path, audio.shape[0], log_f0.shape[0], mel.shape[0])

    return audio, spect, mel, log_f0, energy

def Pattern_File_Generate(path, speaker, emotion, language, gender, dataset, text, decomposed, tag='', eval= False):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path

    file = '{}.{}{}.PICKLE'.format(
        speaker if dataset in speaker else '{}.{}'.format(dataset, speaker),
        '{}.'.format(tag) if tag != '' else '',
        os.path.splitext(os.path.basename(path))[0]
        ).upper()
    if any([
        os.path.exists(os.path.join(x, dataset, speaker, file).replace("\\", "/"))
        for x in [hp.Train.Eval_Pattern.Path, hp.Train.Train_Pattern.Path]
        ]):
        return
    file = os.path.join(pattern_path, dataset, speaker, file).replace("\\", "/")

    audio, spect, mel, log_f0, energy = Pattern_Generate(
        path= path,
        n_fft= hp.Sound.N_FFT,
        num_mels= hp.Sound.Mel_Dim,
        sample_rate= hp.Sound.Sample_Rate,
        hop_size= hp.Sound.Frame_Shift,
        win_size= hp.Sound.Frame_Length,
        fmin= hp.Sound.Mel_F_Min,
        fmax= hp.Sound.Mel_F_Max,
        f0_min= hp.Sound.F0_Min,
        f0_max= hp.Sound.F0_Max,
        top_db= top_db_dict[dataset] if dataset in top_db_dict.keys() else 60
        )
    new_Pattern_dict = {
        'Audio': audio.astype(np.float32),
        'Spectrogram': spect.astype(np.float32),
        'Mel': mel.astype(np.float32),
        'Log_F0': log_f0.astype(np.float32),
        'Energy': energy.astype(np.float32),
        'Speaker': speaker,
        'Emotion': emotion,
        'Language': language,
        'Gender': gender,
        'Dataset': dataset,
        'Text': text,
        'Decomposed': decomposed
        }

    os.makedirs(os.path.join(pattern_path, dataset, speaker).replace('\\', '/'), exist_ok= True)
    with open(file, 'wb') as f:
        pickle.dump(new_Pattern_dict, f, protocol=4)


def Emotion_Info_Load(path):
    '''
    ema, emb, emf, emg, emh, nea, neb, nec, ned, nee, nek, nel, nem, nen, neo
    1-100: Neutral
    101-200: Happy
    201-300: Sad
    301-400: Angry

    lmy, ava, avb, avc, avd, ada, adb, adc, add:
    all neutral
    '''
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            if any(['lmy04282' in file, 'lmy07365' in file]):
                continue
            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    for wav_path in paths:
        text = open(wav_path.replace('/wav/', '/transcript/').replace('.wav', '.txt'), 'r', encoding= 'utf-8-sig').readlines()[0].strip()
        text = Text_Filtering(text)
        if text is None:
            continue

        decomposed = []
        for letter in text:
            if not hgtk.checker.is_hangul(letter):
                decomposed.append(letter)
                continue

            onset, nucleus, coda = hgtk.letter.decompose(letter)
            coda += '_'
            decomposed.extend([onset, nucleus, coda])

        text_dict[wav_path] = text
        decomposed_dict[wav_path] = decomposed

    paths = list(text_dict.keys())

    speaker_dict = {
        path: path.split('/')[-3].strip().upper()
        for path in paths
        }
    
    emotion_dict = {}
    for path in paths:
        if speaker_dict[path] in ['LMY', 'AVA', 'AVB', 'AVC', 'AVD', 'ADA', 'ADB', 'ADC', 'ADD']:
            emotion_dict[path] = 'Neutral'
        elif speaker_dict[path] in ['EMA', 'EMB', 'EMF', 'EMG', 'EMH', 'NEA', 'NEB', 'NEC', 'NED', 'NEE', 'NEK', 'NEL', 'NEM', 'NEN', 'NEO']:
            index = int(os.path.splitext(os.path.basename(path))[0][-5:])
            if index > 0 and index < 101:
                emotion_dict[path] = 'Neutral'
            elif index > 100 and index < 201:
                emotion_dict[path] = 'Happy'
            elif index > 200 and index < 301:
                emotion_dict[path] = 'Sad'
            elif index > 300 and index < 401:
                emotion_dict[path] = 'Angry'
            else:
                raise NotImplementedError('Unknown emotion index: {}'.format(index))
        else:
            raise NotImplementedError('Unknown speaker: {}'.format(speaker_dict[path]))

    language_dict = {path: 'Korean' for path in paths}

    gender_dict = {
        'ADA': 'Female',
        'ADB': 'Female',
        'ADC': 'Male',
        'ADD': 'Male',
        'AVA': 'Female',
        'AVB': 'Female',
        'AVC': 'Female',
        'AVD': 'Female',
        'EMA': 'Female',
        'EMB': 'Female',
        'EMF': 'Male',
        'EMG': 'Male',
        'EMH': 'Male',
        'LMY': 'Female',
        'NEA': 'Female',
        'NEB': 'Female',
        'NEC': 'Female',
        'NED': 'Female',
        'NEE': 'Female',
        'NEK': 'Male',
        'NEL': 'Male',
        'NEM': 'Male',
        'NEN': 'Male',
        'NEO': 'Male',
        }
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in speaker_dict.items()
        }

    print('Emotion info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def KSS_Info_Load(path):
    '''
    all neutral
    '''
    paths, text_dict, decomposed_dict = [], {}, {}
    for line in open(os.path.join(path, 'transcript.v.1.4.txt').replace('\\', '/'), 'r', encoding= 'utf-8-sig').readlines():
        line = line.strip().split('|')
        file, text = line[0].strip(), line[2].strip()
        text = Text_Filtering(text)
        if text is None:
            continue
        decomposed = []
        for letter in text:
            if not hgtk.checker.is_hangul(letter):
                decomposed.append(letter)
                continue

            onset, nucleus, coda = hgtk.letter.decompose(letter)
            coda += '_'
            decomposed.extend([onset, nucleus, coda])

        file = os.path.join(path, 'kss', file).replace('\\', '/')
        paths.append(file)
        text_dict[file] = text
        decomposed_dict[file] = decomposed

    speaker_dict = {
        path: 'KSS'
        for path in paths
        }
    emotion_dict = {
        path: 'Neutral'
        for path in paths
        }
    language_dict = {
        path: 'Korean'
        for path in paths
        }
    gender_dict = {
        path: 'Female'
        for path in paths
        }


    print('KSS info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict, language_dict, gender_dict


def Basic_Info_Load(path, dataset_label, language= None, gender= None):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)
    
    text_dict = {}
    decomposed_dict = {}
    speaker_dict = {}
    emotion_dict = {}

    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8-sig').readlines()[1:]:
        file, text, speaker, emotion = line.strip().split('\t')
        text = Text_Filtering(text)
        if text is None:
            continue
        
        decomposed = Decompose(text)
        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        decomposed_dict[os.path.join(path, file).replace('\\', '/')] = decomposed
        speaker_dict[os.path.join(path, file).replace('\\', '/')] = speaker.strip()
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion.strip()

    paths = list(text_dict.keys())

    language_dict = {path: language for path in paths}

    if type(language) == str or language is None:
        language_dict = {path: language for path in paths}
    elif type(language) == dict:
        language_dict = {
            path: language[speaker]
            for path, speaker in speaker_dict.items()
            }

    if type(gender) == str or gender is None:
        gender_dict = {path: gender for path in paths}
    elif type(gender) == dict:
        gender_dict = {
            path: gender[speaker]
            for path, speaker in speaker_dict.items()
            }


    print('{} info generated: {}'.format(dataset_label, len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict, language_dict, gender_dict


def VCTK_Info_Load(path):
    '''
    VCTK v0.92 is distributed as flac files.
    '''
    path = os.path.join(path, 'wav48').replace('\\', '/')   

    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue

            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    for path in paths:
        if 'p315'.upper() in path.upper():  #Officially, 'p315' text is lost in VCTK dataset.
            continue
        text = Text_Filtering(open(path.replace('wav48', 'txt').replace('flac', 'txt'), 'r').readlines()[0])
        if text is None:
            continue
        text = text.upper()
        
        decomposed = Decompose(text)
        text_dict[path] = text
        decomposed_dict[path] = decomposed
            
    paths = list(text_dict.keys())

    speaker_dict = {
        path: 'VCTK.{}'.format(path.split('/')[-2].strip().upper())
        for path in paths
        }

    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}

    gender_dict = {
        'VCTK.P225': 'Female',
        'VCTK.P226': 'Male',
        'VCTK.P227': 'Male',
        'VCTK.P228': 'Female',
        'VCTK.P229': 'Female',
        'VCTK.P230': 'Female',
        'VCTK.P231': 'Female',
        'VCTK.P232': 'Male',
        'VCTK.P233': 'Female',
        'VCTK.P234': 'Female',
        'VCTK.P236': 'Female',
        'VCTK.P237': 'Male',
        'VCTK.P238': 'Female',
        'VCTK.P239': 'Female',
        'VCTK.P240': 'Female',
        'VCTK.P241': 'Male',
        'VCTK.P243': 'Male',
        'VCTK.P244': 'Female',
        'VCTK.P245': 'Male',
        'VCTK.P246': 'Male',
        'VCTK.P247': 'Male',
        'VCTK.P248': 'Female',
        'VCTK.P249': 'Female',
        'VCTK.P250': 'Female',
        'VCTK.P251': 'Male',
        'VCTK.P252': 'Male',
        'VCTK.P253': 'Female',
        'VCTK.P254': 'Male',
        'VCTK.P255': 'Male',
        'VCTK.P256': 'Male',
        'VCTK.P257': 'Female',
        'VCTK.P258': 'Male',
        'VCTK.P259': 'Male',
        'VCTK.P260': 'Male',
        'VCTK.P261': 'Female',
        'VCTK.P262': 'Female',
        'VCTK.P263': 'Male',
        'VCTK.P264': 'Female',
        'VCTK.P265': 'Female',
        'VCTK.P266': 'Female',
        'VCTK.P267': 'Female',
        'VCTK.P268': 'Female',
        'VCTK.P269': 'Female',
        'VCTK.P270': 'Male',
        'VCTK.P271': 'Male',
        'VCTK.P272': 'Male',
        'VCTK.P273': 'Male',
        'VCTK.P274': 'Male',
        'VCTK.P275': 'Male',
        'VCTK.P276': 'Female',
        'VCTK.P277': 'Female',
        'VCTK.P278': 'Male',
        'VCTK.P279': 'Male',
        'VCTK.P280': 'Female',
        'VCTK.P281': 'Male',
        'VCTK.P282': 'Female',
        'VCTK.P283': 'Male',
        'VCTK.P284': 'Male',
        'VCTK.P285': 'Male',
        'VCTK.P286': 'Male',
        'VCTK.P287': 'Male',
        'VCTK.P288': 'Female',
        'VCTK.P292': 'Male',
        'VCTK.P293': 'Female',
        'VCTK.P294': 'Female',
        'VCTK.P295': 'Female',
        'VCTK.P297': 'Female',
        'VCTK.P298': 'Male',
        'VCTK.P299': 'Female',
        'VCTK.P300': 'Female',
        'VCTK.P301': 'Female',
        'VCTK.P302': 'Male',
        'VCTK.P303': 'Female',
        'VCTK.P304': 'Male',
        'VCTK.P305': 'Female',
        'VCTK.P306': 'Female',
        'VCTK.P307': 'Female',
        'VCTK.P308': 'Female',
        'VCTK.P310': 'Female',
        'VCTK.P311': 'Male',
        'VCTK.P312': 'Female',
        'VCTK.P313': 'Female',
        'VCTK.P314': 'Female',
        'VCTK.P316': 'Male',
        'VCTK.P317': 'Female',
        'VCTK.P318': 'Female',
        'VCTK.P323': 'Female',
        'VCTK.P326': 'Male',
        'VCTK.P329': 'Female',
        'VCTK.P330': 'Female',
        'VCTK.P333': 'Female',
        'VCTK.P334': 'Male',
        'VCTK.P335': 'Female',
        'VCTK.P336': 'Female',
        'VCTK.P339': 'Female',
        'VCTK.P340': 'Female',
        'VCTK.P341': 'Female',
        'VCTK.P343': 'Female',
        'VCTK.P345': 'Male',
        'VCTK.P347': 'Male',
        'VCTK.P351': 'Female',
        'VCTK.P360': 'Male',
        'VCTK.P361': 'Female',
        'VCTK.P362': 'Female',
        'VCTK.P363': 'Male',
        'VCTK.P364': 'Male',
        'VCTK.P374': 'Male',
        'VCTK.P376': 'Male',
        'VCTK.S5': 'Female',
        }
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in speaker_dict.items()
        }

    print('VCTK info generated: {}'.format(len(paths)))

    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def Libri_Info_Load(path):
    gender_path = os.path.join(path, 'Gender.txt').replace('\\', '/')

    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    for path in paths:
        text = Text_Filtering(open('{}.normalized.txt'.format(os.path.splitext(path)[0]), 'r', encoding= 'utf-8-sig').readlines()[0])
        if text is None:
            continue
        text = text.upper()
        
        decomposed = Decompose(text)
        text_dict[path] = text
        decomposed_dict[path] = decomposed

    paths = list(text_dict.keys())

    speaker_dict = {
        path: 'Libri.{:04d}'.format(int(path.split('/')[-3].strip().upper()))
        for path in paths
        }

    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}
    gender_dict = {
        'Libri.{:04d}'.format(int(line.strip().split('\t')[0])): line.strip().split('\t')[1]
        for line in open(gender_path).readlines()[1:]
        }
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in speaker_dict.items()
        }

    print('Libri info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def LJ_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    for line in open(os.path.join(path, 'metadata.csv').replace('\\', '/'), 'r').readlines():
        line = line.strip().split('|')        
        text = Text_Filtering(line[2].strip())
        if text is None:
            continue
        text = text.upper()
        decomposed = Decompose(text)
        wav_path = os.path.join(path, 'wavs', '{}.wav'.format(line[0]))
        
        text_dict[wav_path] = text
        decomposed_dict[wav_path] = decomposed

    paths = list(text_dict.keys())

    speaker_dict = {
        path: 'LJ'
        for path in paths
        }
    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}
    gender_dict = {path: 'Female' for path in paths}

    print('LJ info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict, language_dict, gender_dict


def Split_Eval(paths, eval_ratio= 0.001, min_Eval= 1):
    shuffle(paths)
    index = max(int(len(paths) * eval_ratio), min_Eval)
    return paths[index:], paths[:index]

def Metadata_Generate(eval= False):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path
    metadata_File = hp.Train.Eval_Pattern.Metadata_File if eval else hp.Train.Train_Pattern.Metadata_File

    spectrogram_range_dict = {}
    mel_range_dict = {}
    log_f0_dict = {}
    energy_dict = {}
    speakers = []
    emotions = []
    languages = []
    genders = []
    language_and_gender_dict_by_speaker = {}

    new_Metadata_dict = {
        'N_FFT': hp.Sound.N_FFT,
        'Mel_Dim': hp.Sound.Mel_Dim,
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Frame_Length': hp.Sound.Frame_Length,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Spectrogram_Length_Dict': {},
        'Mel_Length_Dict': {},
        'F0_Length_Dict': {},
        'Energy_Length_Dict': {},
        'Speaker_Dict': {},
        'Emotion_Dict': {},
        'Dataset_Dict': {},
        'File_List_by_Speaker_Dict': {},
        'Text_Length_Dict': {},
        }

    files_TQDM = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_path, followlinks=True)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_path, followlinks=True):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_dict = pickle.load(f)

            file = os.path.join(root, file).replace("\\", "/").replace(pattern_path, '').lstrip('/')

            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Audio', 'Spectrogram', 'Mel', 'Log_F0', 'Energy', 'Speaker', 'Emotion', 'Language', 'Gender', 'Dataset', 'Text', 'Decomposed')
                    ]):
                    continue
                new_Metadata_dict['Audio_Length_Dict'][file] = pattern_dict['Audio'].shape[0]
                new_Metadata_dict['Spectrogram_Length_Dict'][file] = pattern_dict['Spectrogram'].shape[0]
                new_Metadata_dict['Mel_Length_Dict'][file] = pattern_dict['Mel'].shape[0]
                new_Metadata_dict['F0_Length_Dict'][file] = pattern_dict['Log_F0'].shape[0]
                new_Metadata_dict['Energy_Length_Dict'][file] = pattern_dict['Energy'].shape[0]
                new_Metadata_dict['Speaker_Dict'][file] = pattern_dict['Speaker']
                new_Metadata_dict['Emotion_Dict'][file] = pattern_dict['Emotion']
                new_Metadata_dict['Dataset_Dict'][file] = pattern_dict['Dataset']
                new_Metadata_dict['File_List'].append(file)
                if not pattern_dict['Speaker'] in new_Metadata_dict['File_List_by_Speaker_Dict'].keys():
                    new_Metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']] = []
                new_Metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']].append(file)
                new_Metadata_dict['Text_Length_Dict'][file] = len(pattern_dict['Text'])

                if not pattern_dict['Speaker'] in spectrogram_range_dict.keys():
                    spectrogram_range_dict[pattern_dict['Speaker']] = {'Min': math.inf, 'Max': -math.inf}
                if not pattern_dict['Speaker'] in mel_range_dict.keys():
                    mel_range_dict[pattern_dict['Speaker']] = {'Min': math.inf, 'Max': -math.inf}
                if not pattern_dict['Speaker'] in log_f0_dict.keys():
                    log_f0_dict[pattern_dict['Speaker']] = []
                if not pattern_dict['Speaker'] in energy_dict.keys():
                    energy_dict[pattern_dict['Speaker']] = []

                spectrogram_range_dict[pattern_dict['Speaker']]['Min'] = min(spectrogram_range_dict[pattern_dict['Speaker']]['Min'], pattern_dict['Spectrogram'].min().item())
                spectrogram_range_dict[pattern_dict['Speaker']]['Max'] = max(spectrogram_range_dict[pattern_dict['Speaker']]['Max'], pattern_dict['Spectrogram'].max().item())
                mel_range_dict[pattern_dict['Speaker']]['Min'] = min(mel_range_dict[pattern_dict['Speaker']]['Min'], pattern_dict['Mel'].min().item())
                mel_range_dict[pattern_dict['Speaker']]['Max'] = max(mel_range_dict[pattern_dict['Speaker']]['Max'], pattern_dict['Mel'].max().item())

                log_f0_dict[pattern_dict['Speaker']].append(pattern_dict['Log_F0'])
                energy_dict[pattern_dict['Speaker']].append(pattern_dict['Energy'])
                speakers.append(pattern_dict['Speaker'])
                emotions.append(pattern_dict['Emotion'])                
                languages.append(pattern_dict['Language'])
                genders.append(pattern_dict['Gender'])
                language_and_gender_dict_by_speaker[pattern_dict['Speaker']] = {
                    'Language': pattern_dict['Language'],
                    'Gender': pattern_dict['Gender']
                    }
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

            files_TQDM.update(1)

    with open(os.path.join(pattern_path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_dict, f, protocol= 4)

    if not eval:
        yaml.dump(
            spectrogram_range_dict,
            open(hp.Spectrogram_Range_Info_Path, 'w')
            )
        yaml.dump(
            mel_range_dict,
            open(hp.Mel_Range_Info_Path, 'w')
            )

        log_f0_info_dict = {}
        for speaker, log_f0_list in log_f0_dict.items():
            log_f0 = np.hstack(log_f0_list)
            log_f0 = np.clip(log_f0, -10.0, np.inf)
            log_f0 = log_f0[log_f0 != -10.0]

            log_f0_info_dict[speaker] = {
                'Mean': log_f0.mean().item(),
                'Std': log_f0.std().item()
                }
        yaml.dump(
            log_f0_info_dict,
            open(hp.Log_F0_Info_Path, 'w')
            )

        energy_info_dict = {}
        for speaker, energy_list in energy_dict.items():
            energy = np.hstack(energy_list)
            energy_info_dict[speaker] = {
                'Mean': energy.mean().item(),
                'Std': energy.std().item()
                }
        yaml.dump(
            energy_info_dict,
            open(hp.Energy_Info_Path, 'w')
            )

        speaker_index_dict = {
            speaker: index
            for index, speaker in enumerate(sorted(set(speakers)))
            }
        yaml.dump(
            speaker_index_dict,
            open(hp.Speaker_Info_Path, 'w')
            )
            
        emotion_index_dict = {
            emotion: index
            for index, emotion in enumerate(sorted(set(emotions)))
            }
        yaml.dump(
            emotion_index_dict,
            open(hp.Emotion_Info_Path, 'w')
            )

        language_index_dict = {
            language: index
            for index, language in enumerate(sorted(set(languages)))
            }
        yaml.dump(
            language_index_dict,
            open(hp.Language_Info_Path, 'w')
            )

        gender_index_dict = {
            gender: index
            for index, gender in enumerate(sorted(set(genders)))
            }
        yaml.dump(
            gender_index_dict,
            open(hp.Gender_Info_Path, 'w')
            )

        yaml.dump(
            language_and_gender_dict_by_speaker,
            open(hp.Language_and_Gender_Info_by_Speaker_Path, 'w')
            )

    print('Metadata generate done.')

def Token_dict_Generate():
    tokens = \
        ['<S>', '<E>'] + \
        list(hgtk.letter.CHO) + \
        list(hgtk.letter.JOONG) + \
        ['{}_'.format(x) for x in hgtk.letter.JONG] + \
        [chr(x) for x in range(ord('A'), ord('Z') + 1)] + \
        [',', '.', '?', '!', '\'', '-', ' ']
    token_dict = {token: index for index, token in enumerate(tokens)}
    
    os.makedirs(os.path.dirname(hp.Token_Path), exist_ok= True)    
    yaml.dump(token_dict, open(hp.Token_Path, 'w'))

    return token_dict

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-hp", "--hyper_parameters", required=True, type= str)
    argParser.add_argument("-emo", "--emotion_path", required=False)
    argParser.add_argument("-kss", "--kss_path", required=False)
    argParser.add_argument("-yua", "--yua_path", required=False)
    argParser.add_argument("-jps", "--jps_path", required=False)
    argParser.add_argument("-selectstar", "--selectstar_path", required=False)
    argParser.add_argument("-mediazen", "--mediazen_path", required=False)
    argParser.add_argument("-lostark", "--lostark_path", required=False)
    argParser.add_argument("-sea", "--sea_path", required=False)
    argParser.add_argument("-epic7", "--epic7_path", required=False)
    argParser.add_argument("-sghvc", "--sghvc_path", required=False)
    argParser.add_argument("-gp", "--gp_path", required=False)
    argParser.add_argument("-youtube", "--youtube_path", required=False)
    argParser.add_argument("-gcp", "--gcp_path", required=False)
    argParser.add_argument("-clova", "--clova_path", required=False)
    argParser.add_argument("-maum", "--maum_path", required=False)
    argParser.add_argument("-vd", "--voice_drama_path", required=False)

    argParser.add_argument("-vctk", "--vctk_path", required=False)
    argParser.add_argument("-libri", "--libri_path", required=False)
    argParser.add_argument("-lj", "--lj_path", required=False)

    argParser.add_argument("-evalr", "--eval_ratio", default= 0.001, type= float)
    argParser.add_argument("-evalm", "--eval_min", default= 1, type= int)
    argParser.add_argument("-mw", "--max_worker", default= 2, required=False, type= int)

    args = argParser.parse_args()

    global hp
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    train_paths, eval_paths = [], []
    text_dict = {}
    decomposed_dict = {}
    speaker_dict = {}
    emotion_dict = {}
    language_dict = {}
    gender_dict = {}
    dataset_dict = {}
    tag_dict = {}

    if not args.emotion_path is None:
        emotion_paths, emotion_text_dict, emotion_decomposed_dict, emotion_speaker_dict, emotion_emotion_dict, emotion_language_dict, emotion_gender_dict = Emotion_Info_Load(path= args.emotion_path)
        emotion_paths = Split_Eval(emotion_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(emotion_paths[0])
        eval_paths.extend(emotion_paths[1])
        text_dict.update(emotion_text_dict)
        decomposed_dict.update(emotion_decomposed_dict)
        speaker_dict.update(emotion_speaker_dict)
        emotion_dict.update(emotion_emotion_dict)
        language_dict.update(emotion_language_dict)
        gender_dict.update(emotion_gender_dict)
        dataset_dict.update({path: 'Emotion' for paths in emotion_paths for path in paths})
        tag_dict.update({path: '' for paths in emotion_paths for path in paths})

    if not args.kss_path is None:
        kss_paths, kss_text_dict, kss_decomposed_dict, kss_speaker_dict, kss_emotion_dict, kss_language_dict, kss_gender_dict = KSS_Info_Load(path= args.kss_path)
        kss_paths = Split_Eval(kss_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(kss_paths[0])
        eval_paths.extend(kss_paths[1])
        text_dict.update(kss_text_dict)
        decomposed_dict.update(kss_decomposed_dict)
        speaker_dict.update(kss_speaker_dict)
        emotion_dict.update(kss_emotion_dict)
        language_dict.update(kss_language_dict)
        gender_dict.update(kss_gender_dict)
        dataset_dict.update({path: 'KSS' for paths in kss_paths for path in paths})
        tag_dict.update({path: '' for paths in kss_paths for path in paths})

    if not args.yua_path is None:
        yua_paths, yua_text_dict, yua_decomposed_dict, yua_speaker_dict, yua_emotion_dict, yua_language_dict, yua_gender_dict = Basic_Info_Load(
            path= args.yua_path,
            dataset_label= 'YUA',
            language= 'Korean',
            gender= 'Female'
            )        
        yua_paths = Split_Eval(yua_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(yua_paths[0])
        eval_paths.extend(yua_paths[1])
        text_dict.update(yua_text_dict)
        decomposed_dict.update(yua_decomposed_dict)
        speaker_dict.update(yua_speaker_dict)
        emotion_dict.update(yua_emotion_dict)
        language_dict.update(yua_language_dict)
        gender_dict.update(yua_gender_dict)
        dataset_dict.update({path: 'YUA' for paths in yua_paths for path in paths})
        tag_dict.update({path: '' for paths in yua_paths for path in paths})
    
    if not args.jps_path is None:
        jps_paths, jps_text_dict, jps_decomposed_dict, jps_speaker_dict, jps_emotion_dict, jps_language_dict, jps_gender_dict = Basic_Info_Load(
            path= args.jps_path,
            dataset_label= 'JPS',
            language= 'Korean',
            gender= 'Female'
            )
        jps_paths = Split_Eval(jps_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(jps_paths[0])
        eval_paths.extend(jps_paths[1])
        text_dict.update(jps_text_dict)
        decomposed_dict.update(jps_decomposed_dict)
        speaker_dict.update(jps_speaker_dict)
        emotion_dict.update(jps_emotion_dict)
        language_dict.update(jps_language_dict)
        gender_dict.update(jps_gender_dict)
        dataset_dict.update({path: 'JPS' for paths in jps_paths for path in paths})
        tag_dict.update({path: '' for paths in jps_paths for path in paths})

    if not args.selectstar_path is None:
        selectstar_paths, selectstar_text_dict, selectstar_decomposed_dict, selectstar_speaker_dict, selectstar_emotion_dict, selectstar_language_dict, selectstar_gender_dict = Basic_Info_Load(
            path= args.selectstar_path,
            dataset_label= 'SelectStar',
            language= 'Korean',
            gender= {
                'SelectStar_Female_01': 'Female',
                'SelectStar_Female_02': 'Female',
                'SelectStar_Female_03': 'Female',
                'SelectStar_Male_01': 'Male',
                'SelectStar_Male_02': 'Male',
                'SelectStar_Male_03': 'Male',
                'SelectStar_CJY': 'Female',
                'SelectStar_DH': 'Female',
                'SelectStar_KDH': 'Female',
                'SelectStar_KEY': 'Female',
                'SelectStar_KSH': 'Female',
                'SelectStar_KSW': 'Female',
                'SelectStar_LHW': 'Female',
                'SelectStar_LJA': 'Female',
                'SelectStar_SBL': 'Female',
                'SelectStar_SYW': 'Female',
                },
            )
        selectstar_paths = Split_Eval(selectstar_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(selectstar_paths[0])
        eval_paths.extend(selectstar_paths[1])
        text_dict.update(selectstar_text_dict)
        decomposed_dict.update(selectstar_decomposed_dict)
        speaker_dict.update(selectstar_speaker_dict)
        emotion_dict.update(selectstar_emotion_dict)
        language_dict.update(selectstar_language_dict)
        gender_dict.update(selectstar_gender_dict)
        dataset_dict.update({path: 'SelectStar' for paths in selectstar_paths for path in paths})
        tag_dict.update({path: '' for paths in selectstar_paths for path in paths})

    if not args.mediazen_path is None:
        mediazen_paths, mediazen_text_dict, mediazen_decomposed_dict, mediazen_speaker_dict, mediazen_emotion_dict, mediazen_language_dict, mediazen_gender_dict = Basic_Info_Load(
            path= args.mediazen_path,
            dataset_label= 'Mediazen',
            language= 'Korean',
            gender= {
                'Mediazen_KJE': 'Female',
                },
            )
        mediazen_paths = Split_Eval(mediazen_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(mediazen_paths[0])
        eval_paths.extend(mediazen_paths[1])
        text_dict.update(mediazen_text_dict)
        decomposed_dict.update(mediazen_decomposed_dict)
        speaker_dict.update(mediazen_speaker_dict)
        emotion_dict.update(mediazen_emotion_dict)
        language_dict.update(mediazen_language_dict)
        gender_dict.update(mediazen_gender_dict)
        dataset_dict.update({path: 'Mediazen' for paths in mediazen_paths for path in paths})
        tag_dict.update({path: '' for paths in mediazen_paths for path in paths})

    if not args.lostark_path is None:
        lostark_paths, lostark_text_dict, lostark_decomposed_dict, lostark_speaker_dict, lostark_emotion_dict, lostark_language_dict, lostark_gender_dict = Basic_Info_Load(
            path= args.lostark_path,
            dataset_label= 'LostArk',
            language= 'Korean',
            gender= {
                'Sillian': 'Male',
                'Aman': 'Male',
                'Aman_Delain': 'Male',
                'Nia': 'Female',
                'Ninave': 'Female',
                },
            )
        lostark_paths = Split_Eval(lostark_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(lostark_paths[0])
        eval_paths.extend(lostark_paths[1])
        text_dict.update(lostark_text_dict)
        decomposed_dict.update(lostark_decomposed_dict)
        speaker_dict.update(lostark_speaker_dict)
        emotion_dict.update(lostark_emotion_dict)
        language_dict.update(lostark_language_dict)
        gender_dict.update(lostark_gender_dict)
        dataset_dict.update({path: 'LostArk' for paths in lostark_paths for path in paths})
        tag_dict.update({path: '' for paths in lostark_paths for path in paths})

    if not args.sea_path is None:
        sea_paths, sea_text_dict, sea_decomposed_dict, sea_speaker_dict, sea_emotion_dict, sea_language_dict, sea_gender_dict = Basic_Info_Load(
            path= args.sea_path,
            dataset_label= 'SEA',
            language= 'Korean',
            gender= 'Female'
            )
        sea_paths = Split_Eval(sea_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(sea_paths[0])
        eval_paths.extend(sea_paths[1])
        text_dict.update(sea_text_dict)
        decomposed_dict.update(sea_decomposed_dict)
        speaker_dict.update(sea_speaker_dict)
        emotion_dict.update(sea_emotion_dict)
        language_dict.update(sea_language_dict)
        gender_dict.update(sea_gender_dict)
        dataset_dict.update({path: 'SEA' for paths in sea_paths for path in paths})
        tag_dict.update({path: '' for paths in sea_paths for path in paths})
    
    if not args.epic7_path is None:
        epic7_paths, epic7_text_dict, epic7_decomposed_dict, epic7_speaker_dict, epic7_emotion_dict, epic7_language_dict, epic7_gender_dict = Basic_Info_Load(
            path= args.epic7_path,
            dataset_label= 'Epic7',
            language= {
                'Angelica': 'Korean',
                'Angelica_Moonlight': 'Korean',
                'Bellona': 'Korean',
                'Bellona_EN': 'English',
                'Carrot': 'Korean',
                'Carrot_EN': 'English',
                'Cerise': 'Korean',
                'Cerise_EN': 'English',
                'Elena': 'Korean',
                'Elena_EN': 'English',
                'Hazel_EN': 'English',
                'Khawazu_EN': 'English',
                'Kluri_EN': 'English',
                'Lilias': 'Korean',
                'Lilias_EN': 'English',
                'Lorina_EN': 'English',
                'Montmorancy_EN': 'English',
                'Rass_EN': 'English',
                'Ray': 'Korean',
                'Ray_EN': 'English',
                'Ring_Moonlight': 'Korean',
                'Ring_Moonlight_EN': 'English',
                'Roozid_EN': 'English',
                'Sez_Moonlight': 'Korean',
                'Sez_Moonlight_EN': 'English',
                'Vildred_EN': 'English',
                'Vildred_Moonlight_EN': 'English',
                'Yufine_EN': 'English',
                'Tamarinne_Normal': 'Korean',
                'Tamarinne_Songstress': 'Korean',
                },
            gender= {
                'Angelica': 'Female',
                'Angelica_Moonlight': 'Female',
                'Bellona': 'Female',
                'Bellona_EN': 'Female',
                'Carrot': 'Female',
                'Carrot_EN': 'Female',
                'Cerise': 'Female',
                'Cerise_EN': 'Female',
                'Elena': 'Female',
                'Elena_EN': 'Female',
                'Hazel_EN': 'Female',
                'Khawazu_EN': 'Male',
                'Kluri_EN': 'Female',
                'Lilias': 'Female',
                'Lilias_EN': 'Female',
                'Lorina_EN': 'Female',
                'Montmorancy_EN': 'Female',
                'Rass_EN': 'Male',
                'Ray': 'Male',
                'Ray_EN': 'Male',
                'Ring_Moonlight': 'Female',
                'Ring_Moonlight_EN': 'Female',
                'Roozid_EN': 'Male',
                'Sez_Moonlight': 'Male',
                'Sez_Moonlight_EN': 'Male',
                'Vildred_EN': 'Male',
                'Vildred_Moonlight_EN': 'Male',
                'Yufine_EN': 'Female',
                'Tamarinne_Normal': 'Female',
                'Tamarinne_Songstress': 'Female',    
                }
            )
        epic7_paths = Split_Eval(epic7_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(epic7_paths[0])
        eval_paths.extend(epic7_paths[1])
        text_dict.update(epic7_text_dict)
        decomposed_dict.update(epic7_decomposed_dict)
        speaker_dict.update(epic7_speaker_dict)
        emotion_dict.update(epic7_emotion_dict)
        language_dict.update(epic7_language_dict)
        gender_dict.update(epic7_gender_dict)
        dataset_dict.update({path: 'Epic7' for paths in epic7_paths for path in paths})
        tag_dict.update({path: '' for paths in epic7_paths for path in paths})

    if not args.sghvc_path is None:
        sghvc_paths, sghvc_text_dict, sghvc_decomposed_dict, sghvc_speaker_dict, sghvc_emotion_dict, sghvc_language_dict, sghvc_gender_dict = Basic_Info_Load(
            path= args.sghvc_path,
            dataset_label= 'SGHVC',
            language= 'Korean',
            gender= 'Female'
            )
        sghvc_paths = Split_Eval(sghvc_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(sghvc_paths[0])
        eval_paths.extend(sghvc_paths[1])
        text_dict.update(sghvc_text_dict)
        decomposed_dict.update(sghvc_decomposed_dict)
        speaker_dict.update(sghvc_speaker_dict)
        emotion_dict.update(sghvc_emotion_dict)
        language_dict.update(sghvc_language_dict)
        gender_dict.update(sghvc_gender_dict)
        dataset_dict.update({path: 'SGHVC' for paths in sghvc_paths for path in paths})
        tag_dict.update({path: '' for paths in sghvc_paths for path in paths})


    if not args.gp_path is None:
        gp_paths, gp_text_dict, gp_decomposed_dict, gp_speaker_dict, gp_emotion_dict, gp_language_dict, gp_gender_dict = Basic_Info_Load(
            path= args.gp_path,            
            dataset_label= 'GP',
            language= 'Korean'
            )
        gp_paths = Split_Eval(gp_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(gp_paths[0])
        eval_paths.extend(gp_paths[1])
        text_dict.update(gp_text_dict)
        decomposed_dict.update(gp_decomposed_dict)
        speaker_dict.update(gp_speaker_dict)
        emotion_dict.update(gp_emotion_dict)
        language_dict.update(gp_language_dict)
        gender_dict.update(gp_gender_dict)
        dataset_dict.update({path: 'GP' for paths in gp_paths for path in paths})
        tag_dict.update({path: '' for paths in gp_paths for path in paths})

    if not args.youtube_path is None:
        youtube_paths, youtube_text_dict, youtube_decomposed_dict, youtube_speaker_dict, youtube_emotion_dict, youtube_language_dict, youtube_gender_dict = Basic_Info_Load(
            path= args.youtube_path,
            dataset_label= 'Youtube',
            language= 'Korean'
            )
        youtube_paths = Split_Eval(youtube_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(youtube_paths[0])
        eval_paths.extend(youtube_paths[1])
        text_dict.update(youtube_text_dict)
        decomposed_dict.update(youtube_decomposed_dict)
        speaker_dict.update(youtube_speaker_dict)
        emotion_dict.update(youtube_emotion_dict)
        language_dict.update(youtube_language_dict)
        gender_dict.update(youtube_gender_dict)
        dataset_dict.update({path: 'Youtube' for paths in youtube_paths for path in paths})
        tag_dict.update({path: '' for paths in youtube_paths for path in paths})


    if not args.gcp_path is None:
        gcp_paths, gcp_text_dict, gcp_decomposed_dict, gcp_speaker_dict, gcp_emotion_dict, gcp_language_dict, gcp_gender_dict = Basic_Info_Load(
            path= args.gcp_path,
            dataset_label= 'GCP',
            language= {
                'GCP_EN_Female': 'English',
                'GCP_EN_Male': 'English',
                'GCP_KR_Female': 'Korean',
                'GCP_KR_Male': 'Korean'
                },
            gender= {
                'GCP_EN_Female': 'Female',
                'GCP_EN_Male': 'Male',
                'GCP_KR_Female': 'Female',
                'GCP_KR_Male': 'Male'
                },
            )
        gcp_paths = Split_Eval(gcp_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(gcp_paths[0])
        eval_paths.extend(gcp_paths[1])
        text_dict.update(gcp_text_dict)
        decomposed_dict.update(gcp_decomposed_dict)
        speaker_dict.update(gcp_speaker_dict)
        emotion_dict.update(gcp_emotion_dict)
        language_dict.update(gcp_language_dict)
        gender_dict.update(gcp_gender_dict)
        dataset_dict.update({path: 'GCP' for paths in gcp_paths for path in paths})
        tag_dict.update({path: '' for paths in gcp_paths for path in paths})

    if not args.clova_path is None:
        clova_paths, clova_text_dict, clova_decomposed_dict, clova_speaker_dict, clova_emotion_dict, clova_language_dict, clova_gender_dict = Basic_Info_Load(
            path= args.clova_path,
            dataset_label= 'Clova',
            language= {
                'CLOVA_ANNA': 'English',
                'CLOVA_ARA': 'Korean',
                'CLOVA_BORA': 'Korean',
                'CLOVA_CLARA': 'English',
                'CLOVA_DAIN': 'Korean',
                'CLOVA_EUNYOUNG': 'Korean',
                'CLOVA_GOEUN': 'Korean',
                'CLOVA_HAJUN': 'Korean',
                'CLOVA_JAEWOOK': 'Korean',
                'CLOVA_JIHUN': 'Korean',
                'CLOVA_JINHO': 'Korean',
                'CLOVA_JIWON': 'Korean',
                'CLOVA_JIYUN': 'Korean',
                'CLOVA_JOEY': 'English',
                'CLOVA_JONGHYUN': 'Korean',
                'CLOVA_JOONYOUNG': 'Korean',
                'CLOVA_MATT': 'English',
                'CLOVA_MIJIN': 'Korean',
                'CLOVA_MINSANG': 'Korean',
                'CLOVA_MINSEO': 'Korean',
                'CLOVA_SEUNGPYO': 'Korean',
                'CLOVA_SINU': 'Korean',
                'CLOVA_SUJIN': 'Korean',
                'CLOVA_SUNHEE': 'Korean',
                'CLOVA_SUNKYUNG': 'Korean',
                'CLOVA_TAEJIN': 'Korean',
                'CLOVA_WONTAK': 'Korean',
                'CLOVA_YOUNGIL': 'Korean',
                'CLOVA_YUJIN': 'Korean',
                },
            gender= {
                'CLOVA_ANNA': 'Female',
                'CLOVA_ARA': 'Female',
                'CLOVA_BORA': 'Female',
                'CLOVA_CLARA': 'Female',
                'CLOVA_DAIN': 'Female',
                'CLOVA_EUNYOUNG': 'Female',
                'CLOVA_GOEUN': 'Female',
                'CLOVA_HAJUN': 'Male',
                'CLOVA_JAEWOOK': 'Male',
                'CLOVA_JIHUN': 'Male',
                'CLOVA_JINHO': 'Male',
                'CLOVA_JIWON': 'Male',
                'CLOVA_JIYUN': 'Female',
                'CLOVA_JOEY': 'Female',
                'CLOVA_JONGHYUN': 'Male',
                'CLOVA_JOONYOUNG': 'Male',
                'CLOVA_MATT': 'Male',
                'CLOVA_MIJIN': 'Female',
                'CLOVA_MINSANG': 'Male',
                'CLOVA_MINSEO': 'Female',
                'CLOVA_SEUNGPYO': 'Male',
                'CLOVA_SINU': 'Male',
                'CLOVA_SUJIN': 'Female',
                'CLOVA_SUNHEE': 'Female',
                'CLOVA_SUNKYUNG': 'Female',
                'CLOVA_TAEJIN': 'Male',
                'CLOVA_WONTAK': 'Male',
                'CLOVA_YOUNGIL': 'Male',
                'CLOVA_YUJIN': 'Female',
                },
            )
        clova_paths = Split_Eval(clova_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(clova_paths[0])
        eval_paths.extend(clova_paths[1])
        text_dict.update(clova_text_dict)
        decomposed_dict.update(clova_decomposed_dict)
        speaker_dict.update(clova_speaker_dict)
        emotion_dict.update(clova_emotion_dict)
        language_dict.update(clova_language_dict)
        gender_dict.update(clova_gender_dict)        
        dataset_dict.update({path: 'Clova' for paths in clova_paths for path in paths})
        tag_dict.update({path: '' for paths in clova_paths for path in paths})

    if not args.maum_path is None:
        maum_paths, maum_text_dict, maum_decomposed_dict, maum_speaker_dict, maum_emotion_dict, maum_language_dict, maum_gender_dict = Basic_Info_Load(
            path= args.maum_path,
            dataset_label= 'Maum',
            language= {
                'MAUM_BRANDON': 'English',
                'MAUM_FEMALE_CALM': 'Korean',
                'MAUM_FEMALE_HONEST': 'Korean',
                'MAUM_FEMALE_KID': 'Korean',
                'MAUM_FEMALE_NATURAL': 'Korean',
                'MAUM_FEMALE_UNIQUE': 'Korean',
                'MAUM_FEMALE_VIVID': 'Korean',
                'MAUM_MALE_CALM': 'Korean',
                'MAUM_MALE_CAREFUL': 'Korean',
                'MAUM_MALE_FRIENDLY': 'Korean',
                'MAUM_MALE_KID': 'Korean',
                'MAUM_MALE_NATURAL': 'Korean',
                'MAUM_SELENA': 'English',
                },
            gender= {
                'MAUM_BRANDON': 'Male',
                'MAUM_FEMALE_CALM': 'Female',
                'MAUM_FEMALE_HONEST': 'Female',
                'MAUM_FEMALE_KID': 'Female',
                'MAUM_FEMALE_NATURAL': 'Female',
                'MAUM_FEMALE_UNIQUE': 'Female',
                'MAUM_FEMALE_VIVID': 'Female',
                'MAUM_MALE_CALM': 'Male',
                'MAUM_MALE_CAREFUL': 'Male',
                'MAUM_MALE_FRIENDLY': 'Male',
                'MAUM_MALE_KID': 'Male',
                'MAUM_MALE_NATURAL': 'Male',
                'MAUM_SELENA': 'Female',
                },
            )
        maum_paths = Split_Eval(maum_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(maum_paths[0])
        eval_paths.extend(maum_paths[1])
        text_dict.update(maum_text_dict)
        decomposed_dict.update(maum_decomposed_dict)
        speaker_dict.update(maum_speaker_dict)
        emotion_dict.update(maum_emotion_dict)
        language_dict.update(maum_language_dict)
        gender_dict.update(maum_gender_dict)
        dataset_dict.update({path: 'Maum' for paths in maum_paths for path in paths})
        tag_dict.update({path: '' for paths in maum_paths for path in paths})

    if not args.voice_drama_path is None:
        voice_drama_paths, voice_drama_text_dict, voice_drama_decomposed_dict, voice_drama_speaker_dict, voice_drama_emotion_dict, voice_drama_language_dict, voice_drama_gender_dict = Basic_Info_Load(
            path= args.voice_drama_path,
            dataset_label= 'VD',
            language= 'Korean',
            gender= 'Female',
            )
        voice_drama_paths = Split_Eval(voice_drama_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(voice_drama_paths[0])
        eval_paths.extend(voice_drama_paths[1])
        text_dict.update(voice_drama_text_dict)
        decomposed_dict.update(voice_drama_decomposed_dict)
        speaker_dict.update(voice_drama_speaker_dict)
        emotion_dict.update(voice_drama_emotion_dict)
        language_dict.update(voice_drama_language_dict)
        gender_dict.update(voice_drama_gender_dict)
        dataset_dict.update({path: 'VD' for paths in voice_drama_paths for path in paths})
        tag_dict.update({path: '' for paths in voice_drama_paths for path in paths})


    if not args.vctk_path is None:
        vctk_paths, vctk_text_dict, vctk_decomposed_dict, vctk_speaker_dict, vctk_emotion_dict, vctk_language_dict, vctk_gender_dict = VCTK_Info_Load(path= args.vctk_path)
        vctk_paths = Split_Eval(vctk_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(vctk_paths[0])
        eval_paths.extend(vctk_paths[1])
        text_dict.update(vctk_text_dict)
        decomposed_dict.update(vctk_decomposed_dict)
        speaker_dict.update(vctk_speaker_dict)
        emotion_dict.update(vctk_emotion_dict)
        language_dict.update(vctk_language_dict)
        gender_dict.update(vctk_gender_dict)
        dataset_dict.update({path: 'VCTK' for paths in vctk_paths for path in paths})
        tag_dict.update({path: '' for paths in vctk_paths for path in paths})

    if not args.libri_path is None:
        libri_paths, libri_text_dict, libri_decomposed_dict, libri_speaker_dict, libri_emotion_dict, libri_language_dict, libri_gender_dict = Libri_Info_Load(path= args.libri_path)
        libri_paths = Split_Eval(libri_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(libri_paths[0])
        eval_paths.extend(libri_paths[1])
        text_dict.update(libri_text_dict)
        decomposed_dict.update(libri_decomposed_dict)
        speaker_dict.update(libri_speaker_dict)
        emotion_dict.update(libri_emotion_dict)
        language_dict.update(libri_language_dict)
        gender_dict.update(libri_gender_dict)
        dataset_dict.update({path: 'Libri' for paths in libri_paths for path in paths})
        tag_dict.update({path: '' for paths in libri_paths for path in paths})

    if not args.lj_path is None:
        lj_paths, lj_text_dict, lj_decomposed_dict, lj_speaker_dict, lj_emotion_dict, lj_language_dict, lj_gender_dict = LJ_Info_Load(path= args.lj_path)
        lj_paths = Split_Eval(lj_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(lj_paths[0])
        eval_paths.extend(lj_paths[1])
        text_dict.update(lj_text_dict)
        decomposed_dict.update(lj_decomposed_dict)
        speaker_dict.update(lj_speaker_dict)
        emotion_dict.update(lj_emotion_dict)
        language_dict.update(lj_language_dict)
        gender_dict.update(lj_gender_dict)
        dataset_dict.update({path: 'LJ' for paths in lj_paths for path in paths})
        tag_dict.update({path: '' for paths in lj_paths for path in paths})

    # if len(train_paths) == 0 or len(eval_paths) == 0:
    #     raise ValueError('Total info count must be bigger than 0.')

    token_dict = Token_dict_Generate()

    with PE(max_workers = args.max_worker) as pe:
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [
                    (
                        path,
                        speaker_dict[path],
                        emotion_dict[path],
                        language_dict[path],
                        gender_dict[path],
                        dataset_dict[path],
                        text_dict[path],
                        decomposed_dict[path],
                        tag_dict[path],
                        False
                        )
                    for path in train_paths
                    ]
                ),
            total= len(train_paths)
            ):
            pass
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [
                    (
                        path,
                        speaker_dict[path],
                        emotion_dict[path],
                        language_dict[path],
                        gender_dict[path],
                        dataset_dict[path],
                        text_dict[path],
                        decomposed_dict[path],
                        tag_dict[path],
                        True
                        )
                    for path in eval_paths
                    ]
                ),
            total= len(eval_paths)
            ):
            pass

    Metadata_Generate()
    Metadata_Generate(eval= True)


# python Pattern_Generator.py -hp Hyper_Parameters.yaml \
#     -emo /datasets/rawdata/Emotion \
#     -kss /datasets/rawdata/KSS \ /datasets/rawdata/Yua_A \
#     -epic7 /datasets/rawdata/222 \
#     -gcp /datasets/rawdata/Dataset_Generator_from_External/External/GCP_TTS \
#     -clova /datasets/rawdata/Dataset_Generator_from_External/External/Clova \
#     -maum /datasets/rawdata/Dataset_Generator_from_External/External/Maum \
#     -selectstar /datasets/rawdata/SelectStar \
#     -mediazen /datasets/rawdata/Mediazen \
#     -lostark /datasets/rawdata/LostArk \
#     -sea /datasets/rawdata/Sea \
#     -vd /datasets/rawdata/Voice_Drama \
#     -vctk /datasets/rawdata/VCTK \
#     -libri /datasets/rawdata/LibriTTS/train \
#     -lj /datasets/rawdata/LJSpeech-1.1 \
#     -sghvc /datasets/rawdata/SGHVC \
#     -evalm 3

# python Pattern_Generator.py -hp Hyper_Parameters.yaml -emo E:/emotions -sghvc E:/SGHVC -evalm 3
# python Pattern_Generator.py -hp Hyper_Parameters.yaml \
#     -epic7 /datasets/rawdata/Epic_Seven \
#     -gcp /datasets/rawdata/Dataset_Generator_from_External/External/GCP_TTS \
#     -clova /datasets/rawdata/Dataset_Generator_from_External/External/Clova \
#     -maum /datasets/rawdata/Dataset_Generator_from_External/External/Maum \
#     -evalm 3