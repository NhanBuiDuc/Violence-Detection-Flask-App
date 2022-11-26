import threading
import Feature_Extractor.option as option
from Feature_Extractor.models.i3d.extract_i3d import ExtractI3D
import logging
import torch
import cv2
import numpy as np
from Violence_Detection.infer import infer
import wave
import time
from datetime import datetime
from Queue import Queue as CustomQueue
from queue import Queue

def read_i3d_demo(video_queue, extract_event):
    extract_event.wait()
    torch.cuda.empty_cache()
    args = option.parser.parse_args()
    
    I3D_extractor = ExtractI3D(args)

    video_features = I3D_extractor.extract_demo(video_queue, extract_event)
    #return
    queue.enque(video_features)


def read_i3d(interval_extract_event, frames_queue, i3d_queue):
    interval_extract_event.wait()
    torch.cuda.empty_cache()
    args = option.parser.parse_args()
    
    I3D_extractor = ExtractI3D(args)

    video_features = I3D_extractor.extract(interval_extract_event, frames_queue, i3d_queue)
    #return
    queue.enque(video_features)

def extract_demo(video_queue: Queue, extract_event):

        global queue
        queue = CustomQueue()

        read_i3d_thread = threading.Thread(target=read_i3d_demo, args=(video_queue, extract_event, ))
                
        read_i3d_thread.start()

        read_i3d_thread.join()

        feature_frame = queue.dequeue()

        return feature_frame

def extract(interval_extract_event, frames_queue, i3d_queue):

        global queue
        queue = CustomQueue()

        read_i3d_thread = threading.Thread(target=read_i3d, args=(interval_extract_event, frames_queue, i3d_queue))
                
        read_i3d_thread.start()

        read_i3d_thread.join()

        feature_frame = queue.dequeue()

        return feature_frame

def listener(interval_extract_event, frames_queue, i3d_queue):
    
    interval_extract_event.wait()
    return extract(interval_extract_event, frames_queue, i3d_queue)


def listener_demo(video_queue: Queue, extract_event):
    
    extract_event.wait()
    return extract_demo(video_queue, extract_event)