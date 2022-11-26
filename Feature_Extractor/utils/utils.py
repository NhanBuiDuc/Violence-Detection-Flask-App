import os
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Union
import platform

import numpy as np
import torch.nn.functional as F

IMAGENET_CLASS_PATH = 'Feature_Extractor/utils/IN_label_map.txt'
KINETICS_CLASS_PATH = 'Feature_Extractor/utils/K400_label_map.txt'

def make_path(output_root, video_path, output_key, ext):
    # extract file name and change the extention
    fname = f'{Path(video_path).stem}_{output_key}{ext}'
    # construct the paths to save the features
    return os.path.join(output_root, fname)

def form_slices(size: int, stack_size: int, step_size: int) -> list((int, int)):
    '''print(form_slices(100, 15, 15) - example'''
    slices = []
    # calc how many full stacks can be formed out of framepaths
    full_stack_num = (size - stack_size) // step_size + 1
    for i in range(full_stack_num):
        start_idx = i * step_size
        end_idx = start_idx + stack_size
        slices.append((start_idx, end_idx))
    return slices


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    # Determine the platform on which the program is running
    if platform.system().lower() == 'windows':
        result = subprocess.run(['where', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ffmpeg_path = result.stdout.decode('utf-8').replace('\r\n', '')
    else:
        result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path


def extract_wav_from_mp4(video_path: str, tmp_path: str) -> str:
    '''Extracts .wav file from .aac which is extracted from .mp4
    We cannot convert .mp4 to .wav directly. For this we do it in two stages: .mp4 -> .aac -> .wav

    Args:
        video_path (str): Path to a video
        audio_path_wo_ext (str):

    Returns:
        [str, str] -- path to the .wav and .aac audio
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # extract video filename from the video_path
    video_filename = os.path.split(video_path)[-1].replace('.mp4', '')

    # the temp files will be saved in `tmp_path` with the same name
    audio_aac_path = os.path.join(tmp_path, f'{video_filename}.aac')
    audio_wav_path = os.path.join(tmp_path, f'{video_filename}.wav')
    
    # constructing shell commands and calling them
    
    mp4_to_acc = f'{which_ffmpeg()} -hide_banner -loglevel panic -y -i {video_path} -acodec copy {audio_aac_path}'
    aac_to_wav = f'{which_ffmpeg()} -hide_banner -loglevel panic -y -i {audio_aac_path} {audio_wav_path}'
    subprocess.call(mp4_to_acc.split())
    subprocess.call(aac_to_wav.split())
    print("done")
    return audio_wav_path, audio_aac_path


def build_cfg_path(feature_type: str) -> os.PathLike:
    '''Makes a path to the default config file for each feature family.

    Args:
        feature_type (str): the type (e.g. 'vggish')

    Returns:
        os.PathLike: the path to the default config for the type
    '''
    path_base = Path('./configs')
    path = path_base / f'{feature_type}.yml'
    return path


def dp_state_to_normal(state_dict):
    '''Converts a torch.DataParallel checkpoint to regular'''
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module'):
            new_state_dict[k.replace('module.', '')] = v
    return new_state_dict


def load_numpy(fpath):
    return np.load(fpath)

def write_numpy(fpath, value):
    return np.save(fpath, value)

def load_pickle(fpath):
    return pickle.load(open(fpath, 'rb'))

def write_pickle(fpath, value):
    return pickle.dump(value, open(fpath, 'wb'))
