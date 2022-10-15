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


CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
WAVE_OUTPUT_FILENAME = "output.wav"

def main():
    torch.cuda.empty_cache()
    args1 = option.parser.parse_args()
    args2 = option.parser.parse_args()
    if args1.on_extraction in ['save_numpy', 'save_pickle']:
        print(f'Saving features to {args1.output_path}')
    if args2.on_extraction in ['save_numpy', 'save_pickle']:
        # args2.output_path = 'output-VGGish'
        print(f'Saving features to {args2.output_path}')
    print('Device:', args1.device)

    # import are done here to avoid import errors (we have two conda environements)

    I3D_extractor = ExtractI3D(args1)
    args2.feature_type = 'vggish'
    VGGish_extractor = ExtractVGGish(args2)

    # # yep, it is this simple!
    # 1.Get frame from camera

    # define a video capture object
    caption = cv2.VideoCapture(0)
    first_frame = True
    rgb_stack = []
    rgb_features_dict = {'rgb': []}
    rgb_stack_counter = 0
    is_recording = False
    is_starting = False

    iteration = 0
    while(True):
        if is_starting == False:
            start = datetime.now()

            start_time = start.strftime("%H:%M:%S")
            print("Start Time =", start_time)
            is_starting = True
        iteration += 1
        print(iteration)
        # Get current run time
        now = datetime.now() # time object
        print("Current =", now - start, " s")
        
        # Start recording
        if is_recording == False:
            # Capture audio
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

            print("* recording")
            is_recording = True
            frames = []

        data = stream.read(CHUNK)
        frames.append(data)

        # Capture the video frame
        # by frame
        frame_exists, rgb = caption.read()
       
        # 2. Extract I3D, VGGish

        # return a features of n * 1048 for each 64 frame
        rgb_features = I3D_extractor._extract(rgb, rgb_stack)
        #VGGish_extractor
        if rgb_features != None:
            rgb_features_dict['rgb'].append(rgb_features['rgb'].tolist())
            rgb_stack = []
            rgb_stack_counter += 1
        print("Stack counter", rgb_stack_counter)

        # Stop the extraction
        if rgb_stack_counter == args1.stack_size:
            # transforms list of features into a np array
            rgb_features_dict = np.array(rgb_features_dict['rgb'], dtype=np.float32)
            rgb_features_dict_feeded = rgb_features_dict.squeeze(1)
            RECORD_SECONDS = (now - start).seconds

            # 3. Send to model To XD model
            #infer(rgb_features_dict_feeded)
            rgb_features_dict = {'rgb': []}
            rgb_stack_counter = 0
            
        # 4. Save a video of violence to a file
        
        # Display the resulting frame
        cv2.imshow('frame', rgb)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    caption.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
