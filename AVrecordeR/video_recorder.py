from datetime import datetime as DateTime
import datetime
import cv2
import wave
import threading
import time
from frame import Frame
class VideoRecorder():
    
    # Video class based on openCV 
    def __init__(self, video_filename = None):
        self.open = True
        self.video_filename = video_filename
        self.device_index = 0
        self.video_cap = cv2.VideoCapture(self.video_filename)
        self.webcam_cap = cv2.VideoCapture(self.device_index)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = self.webcam_cap.get(cv2.CAP_PROP_FPS)
        self.save_interval = 5
        self.frame_count = 0
        self.frameSize = (640,480)
        
    # Video starts being recorded 
    def record(self, single_frame_event, interval_extract_event, process_stop,  outputRGBs, frames_queue, outputRGB):  
                reset = False     
                frame = Frame(rgb = [])
                frame.start = DateTime.now()
                print("Start", frame.start)
                while(process_stop.isSet() == True):
                    ret, video_frame = self.video_cap.read()

                    if (ret == True):

                        # # Display the resulting frame
                        # cv2.imshow('Video', video_frame)
                    
                        # # Press Q on keyboard to  exit
                        # if cv2.waitKey(25) & 0xFF == ord('q'):
                        #     break

                        single_frame_event.set()
                        # if self.frame_count == 0 and reset == True:
                        #     frame = Frame(rgb = [])
                        #     frame.start = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                        #     reset = False
                            
                        if(frame != None):
                            frame.rgb.append(video_frame)

                        self.frame_count += 1
                        if (self.frame_count == (self.fps * self.save_interval)):
                            frame.end = (frame.start + datetime.timedelta(0, 5))
                            newStartTime = frame.end
                            print("End", frame.end)
                            
                            self.frame_count = 0
                            # reset = True
                            frames_queue.append(frame)
                            frame = Frame(rgb = [])
                            frame.start = newStartTime
                            # reset = False
                        outputRGB = video_frame
                        outputRGBs.append(video_frame)
                        
                        if len(frames_queue) > 5:
                            interval_extract_event.set()
                    else:
                        self.stop()
                        break

    # Video starts being recorded 
    def record_webcam_demo(self, frame_queue, extract_event):
      
        while(self.open==True):
            ret, video_frame = self.webcam_cap.read()

            if (ret == True):

                # Display the resulting frame
                cv2.imshow('Video', video_frame)
            
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                if self.frame_count == 0:
                    frame = Frame(start = datetime.now(), rgb = [])

                frame.rgb.append(video_frame)

                self.frame_count += 1
                # Duration : 60s for 24fps, means frame_count = fps * duration = 24 *  60
                # Recorded at 30 fps, but wanted_fps is 24, the ratio 24/30 = 0.8.
                if (self.frame_count == (self.fps * self.save_interval)):
                    frame.end = datetime.now()
                    self.frame_count = 0
                    frame_queue.put(frame)

                if not frame_queue.empty():
                    extract_event.set()   
                
            else:
                self.stop()
                break               
            
    # Video starts being recorded 
    def record_video_demo(self, frame_queue, extract_event):
        
        while(self.open==True):
            ret, video_frame = self.video_cap.read()

            if (ret == True):
                # Display the resulting frame
                cv2.imshow('Video', video_frame)
            
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                if self.frame_count == 0:
                    frame = Frame(start = datetime.now(), rgb = [])

                frame.rgb.append(video_frame)
                
                self.frame_count += 1
                # Duration : 60s for 24fps, means frame_count = fps * duration = 24 *  60
                # Recorded at 30 fps, but wanted_fps is 24, the ratio 24/30 = 0.8.
                if (self.frame_count == (self.fps * self.save_interval)):
                    frame.end = datetime.now()
                    self.frame_count = 0
                    frame_queue.put(frame)

                if not frame_queue.empty():
                    extract_event.set()   
                
            else:
                self.stop()
                break

    # Finishes the video recording therefore the thread too
    def stop(self):
        
        if self.open==True:
            
            self.open=False
            # self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()
            
        else: 
            pass


    # # Launches the video recording function using a thread			
    # def start(self, stream_queue, frame_queue, wanted_fps, extract_event, stream_event):
    #     video_thread = threading.Thread(target=self.record, args=(stream_queue, frame_queue, wanted_fps, extract_event, stream_event, ))
    #     video_thread.start()

    # # Launches the video recording function using a thread			
    # def start_video(self, stream_queue, frame_queue, wanted_fps, extract_event, stream_event):
    #     video_thread = threading.Thread(target=self.record_video, args=(stream_queue, frame_queue, wanted_fps, extract_event, stream_event, ))
    #     video_thread.start()
    
    # Launches the video recording function using a thread			
    def start_video_demo(self, frame_queue, extract_event):
        video_thread = threading.Thread(target=self.record_video_demo, args=(frame_queue, extract_event, ))
        video_thread.start()
    
    # Launches the video recording function using a thread			
    def start_webcam_demo(self, frame_queue, extract_event):
        video_thread = threading.Thread(target=self.record_webcam_demo, args=(frame_queue, extract_event, ))
        video_thread.start()