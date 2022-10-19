import cv2
import threading
import time
import subprocess
import os
from AVrecordeR.video_recorder import VideoRecorder
from AVrecordeR.audio_recorder import AudioRecorder
########################
## JRF
## VideoRecorder and AudioRecorder are two classes based on openCV and pyaudio, respectively. 
## By using multithreading these two classes allow to record simultaneously video and audio.
## ffmpeg is used for muxing the two signals
## A timer loop is used to control the frame rate of the video recording. This timer as well as
## the final encoding rate can be adjusted according to camera capabilities
##

########################
## Usage:
## 
## numpy, PyAudio and Wave need to be installed
## install openCV, make sure the file cv2.pyd is located in the same folder as the other libraries
## install ffmpeg and make sure the ffmpeg .exe is in the working directory
##
## 
## start_AVrecording(filename) # function to start the recording
## stop_AVrecording(filename)  # "" ... to stop it
##
##
########################


# Required and wanted processing of final files
def file_manager(filename):

	local_path = os.getcwd()

	if os.path.exists(str(local_path) + "/temp_audio.wav"):
		os.remove(str(local_path) + "/temp_audio.wav")
	
	if os.path.exists(str(local_path) + "/temp_video.avi"):
		os.remove(str(local_path) + "/temp_video.avi")

	if os.path.exists(str(local_path) + "/temp_video2.avi"):
		os.remove(str(local_path) + "/temp_video2.avi")

	if os.path.exists(str(local_path) + "/" + filename + ".avi"):
		os.remove(str(local_path) + "/" + filename + ".avi")

def start_AVrecording(filename):
				
	global video_thread
	global audio_thread
	
	# not threads but objects
	video_thread = VideoRecorder()
	audio_thread = AudioRecorder()

	audio_thread.start()
	video_thread.start()

	return filename

def start_video_recording(filename):
				
	global video_thread
	
	video_thread = VideoRecorder()
	video_thread.start()

	return filename
	

def start_audio_recording(filename):
				
	global audio_thread
	
	audio_thread = AudioRecorder()
	audio_thread.start()

	return filename

def stop_AVrecording(filename):
	
	audio_thread.stop() 
	frame_counts = video_thread.frame_counts
	elapsed_time = time.time() - video_thread.start_time
	recorded_fps = frame_counts / elapsed_time
	print ("total frames ",str(frame_counts))
	print ("elapsed time ", str(elapsed_time))
	print ("recorded fps ", str(recorded_fps))
	video_thread.stop() 

	# Makes sure the threads have finished
	while threading.active_count() > 1:
		time.sleep(1)

	
#	 Merging audio and video signal
	
	if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected
										
		print ("Re-encoding")
		cmd = "ffmpeg -r " + str(recorded_fps) + " -i temp_video.avi -pix_fmt yuv420p -r 6 temp_video2.avi"
		subprocess.call(cmd, shell=True)
	
		print ("Muxing")
		cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
		subprocess.call(cmd, shell=True)
	
	else:
		
		print ("Normal recording\nMuxing")
		cmd = "AVrecorderR/ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
		subprocess.call(cmd, shell=True)

		print ("..")	

	
def record():
	
	filename = "Default_user"	
	file_thread = threading.Thread(target=file_manager(filename))
	
	file_thread.start()

	start_AVrecording(filename)  
	#15s recording
	time.sleep(5)
		
	stop_AVrecording(filename)
	print ("Recording Done")
	return filename+".avi", "temp_audio.wav"