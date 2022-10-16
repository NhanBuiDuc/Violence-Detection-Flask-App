import threading
import Feature_Extractor.option as option
from Feature_Extractor.utils.utils import form_list_from_user_input
from Feature_Extractor.models.i3d.extract_i3d import ExtractI3D
from Feature_Extractor.models.vggish.extract_vggish import ExtractVGGish
import os
import torch
import cv2
import numpy as np
from Violence_Detection.infer import infer
import pyaudio
import wave
import time
from datetime import datetime
from Queue import Queue

def read_i3d(filename):

    torch.cuda.empty_cache()
    args = option.parser.parse_args()
    
    I3D_extractor = ExtractI3D(args)

    video_features = I3D_extractor.extract(filename)
    #return
    queue.enque(video_features)

def read_vggish(filename):
    args = option.parser.parse_args()
    args.feature_type = 'vggish'
    VGGish_extractor = ExtractVGGish(args)
    audio_features = VGGish_extractor.extract(filename)
    # return
    queue.enque(audio_features)

def extract(video_file, audio_file):
    global queue
    queue = Queue()
    read_i3d_thread = threading.Thread(target=read_i3d, args=(video_file,))
    read_i3d_thread.start()
    read_i3d_thread.join()

    read_vggish_thread = threading.Thread(target=read_vggish, args=(audio_file,))
    read_vggish_thread.start()
    read_vggish_thread.join()


    video_features= queue.dequeue()
    audio_features = queue.dequeue()

    np_video_features = np.array(video_features['rgb'], dtype=np.float32)
    np_audio_features = np.array(audio_features['vggish'], dtype=np.float32)

    return np_video_features, np_audio_features