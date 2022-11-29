import numpy as np
import multiprocessing
from multiprocessing import Manager
import pickle
import cv2
from AVrecordeR.video_recorder import VideoRecorder
import Violence_Detection.infer as detector
import Feature_Extractor.option as i3d_option
from Feature_Extractor.models.i3d.extract_i3d import ExtractI3D
from prediction import Prediction
import threading
import Feature_Extractor.extractor as extractor
import time
class StreamingInstance():
     def __init__(self, id, video_filename = None):
          self.id = id
          self.video_filename = video_filename
          self.XDmodel = detector.load_model()
          
          self.recorder = VideoRecorder(video_filename=self.video_filename)

          self.SSDmodel = pickle.load(open('SSD/models/model.pkl', 'rb'))
          self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                         "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                         "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                         "sofa", "train", "tvmonitor"]
          self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
          self.net = cv2.dnn.readNetFromCaffe("SSD/MobileNetSSD_deploy.prototxt", "SSD/MobileNetSSD_deploy.caffemodel")
          self.single_frame_event = threading.Event()
          self.interval_extract_event = threading.Event()
          self.process_running = threading.Event()
          
          manager = Manager()
          self.manager = manager.dict()
          self.manager['i3d_queue'] = []
          self.manager['frames_queue'] = []
          self.manager['outputRGBs'] = []
          self.manager['outputRGB'] = []
          # A list of frame also containing i3d feature
          self.i3d_queue = self.manager['i3d_queue']
          # The queue of frames corresponding to the saving interval, containing the start, end, and RGB frames needed to be processed in batch.
          self.frames_queue = self.manager['frames_queue']
          # List of frames recorded in real time
          self.outputRGBs = self.manager['outputRGBs']
          # Current RGB frame picked from frame list
          self.outputRGB = self.manager['outputRGB']

          self.ssd_lock = threading.Lock()
          self.i3d_lock = threading.Lock()
          self.frame_count = 0
          self.fps = self.recorder.fps
          self.save_interval = 5
          self.xd_batch = 5
          self.i3d_args = i3d_option.parser.parse_args()
          self.i3d_extractor = ExtractI3D(self.i3d_args)
          self.prediction = Prediction(thresh_hold=0.5)
          self.record_process = None
          self.ssd_process = None
          self.xd_process = None
          self.extract_process = None
          self.stream_process = None
     
     def start(self):
          self.process_running.set()
          record_process = threading.Thread(target=self.recorder.record, args = ( self.single_frame_event, self.interval_extract_event, self.process_running , self.outputRGBs, self.frames_queue, self.outputRGB,))
          ssd_process = threading.Thread(target=self.ssd, args=())

          xd_process = threading.Thread(target=self.xd, args=())
          # stream_process = threading.Thread(target=self.streaming, args=())

          # extract_process.start()
          record_process.start()
          xd_process.start()
          ssd_process.start()
          # stream_process.start()
          
          self.record_process = record_process
          self.xd_process = xd_process
          # self.stream_process = stream_process
          self.ssd_process = ssd_process

     def get_video_source(self):
          id = self.id
          if(id == 1):
               self.video_filename = "demo.mp4"
          elif(id == 2):
               self.video_filename = "demo2.mp4"
          elif(id == 3):
               self.video_filename = "demo3.mp4"
          elif(id == 4):
               self.video_filename = "demo4.mp4"
     def xd(self):
          self.interval_extract_event.wait()
          while(self.process_running.isSet() == True):
               time.sleep(0.5)
               if(len(self.frames_queue) > 0):
                    self.i3d_extractor.extract(self.interval_extract_event, self.frames_queue, self.i3d_queue)
               if(len(self.i3d_queue) >= 5):
                    detector.inference(frames = self.i3d_queue, model=self.XDmodel, batch = self.xd_batch, prediction = self.prediction)
                    print("At: ", self.prediction.start)
                    print("Violence: ", self.prediction.prediction)
                    print("Violent Rate: ", self.prediction.score)
                    self.i3d_queue.pop(0)

     def get_prediction(self):
          if self.prediction != None:
               return self.prediction
          else:
               return None
     def raw_stream(self):
          while(self.process_running.isSet() == True):
               if self.outputRGB is None:
                    continue
               if(len(self.outputRGB) > 0):
                    (flag, encodedImage) = cv2.imencode(".jpg", self.outputRGB)
                    if not flag:
                         continue
                    # yield the output frame in the byte format
                    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                         bytearray(encodedImage) + b'\r\n')
     def streaming(self):
          self.single_frame_event.wait()
          while(True):
               if(len(self.outputRGBs) > 0):
                    with self.ssd_lock:
                         self.outputRGB = self.outputRGBs.pop(0)
     def ssd(self):
               self.single_frame_event.wait()
               while(self.process_running.isSet() == True):
                    # Preprossesing the frames to obtain detections
                    if (len(self.outputRGBs)) != 0:
                         blob, frame = self.preprocess_video_frames(self.outputRGBs.pop(0))

                         # Obtaining detections from each frame
                         detections = self.get_detections_from_frames(blob)
                         
                         # loop over the detections
                         for key in np.arange(0, detections.shape[2]):
                              # extract the confidence associated with the prediction
                              confidence = detections[0, 0, key, 2]

                              # filter out weak detections
                              idx = self.filter_out_detections(detections, confidence, key)

                              if(idx == None):
                                   break
                              else:
                                   if(idx == 15):
                                        # Get the bounding boxes for the detections
                                        box = self.draw_bounding_box(detections, frame, key)
                                        
                                        # Get predictions for the boxes
                                        label = self.predict_class_labels(confidence, idx)

                                        # Draw predictions on frames
                                        frame = self.draw_predictions_on_frames(box, label, frame, idx)
                         with self.ssd_lock:
                              self.outputRGB = frame.copy()
                    self.single_frame_event.wait()

     def preprocess_video_frames(self, frame):

          blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
               0.007843, (300, 300), 127.5)
          return blob, frame

     def get_detections_from_frames(self, blob):
          """pass the blob through the network and obtain the detections and
          predictions"""
          self.net.setInput(blob)
          detections = self.net.forward()
          return detections

     def filter_out_detections(self, detections, confidence, key):
          if confidence > 0.2:
               # extract the index of the class label from the
               # `detections`, then compute the (x, y)-coordinates of
               # the bounding box for the object
               idx = int(detections[0, 0, key, 1])
          else:
               idx = None
          return idx

     def draw_bounding_box(self, detections, frame, key):
          """Draw bounding boxes for the filtered detections"""
          # grab the frame dimensions
          (h, w) = frame.shape[:2]
          # get coordinates for bounding boxes
          box = detections[0, 0, key, 3:7] * np.array([w, h, w, h])
          return box
     def predict_class_labels(self, confidence, idx):
          # Get labels for the detections
          label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)
          return label
               
     def draw_predictions_on_frames(self, box, label, frame, idx):
          (startX, startY, endX, endY) = box.astype("int")
          # draw the prediction on the frame
          cv2.rectangle(frame, (startX, startY), (endX, endY), self.COLORS[idx], 2)
          y = startY - 15 if startY - 15 > 15 else startY + 15
          cv2.putText(frame, label, (startX, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
          return frame

     def generate(self):
          while self.process_running.isSet() == True:
                    # wait until the lock is acquired
                    with self.ssd_lock:
                         # check if the output frame is available, otherwise skip
                         # the iteration of the loop
                         if self.outputRGB is None:
                              continue
                         # encode the frame in JPEG format
                         (flag, encodedImage) = cv2.imencode(".jpg", self.outputRGB)
                         # ensure the frame was successfully encoded
                         if not flag:
                              continue
                    # yield the output frame in the byte format
                    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                         bytearray(encodedImage) + b'\r\n')

     def stop(self):
          self.process_running.clear()
          self.recorder.stop()
          return("ENDED THE STREAM")