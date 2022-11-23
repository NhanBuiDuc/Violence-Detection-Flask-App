from queue import Queue
import AVrecordeR.recorder as recorder
import Feature_Extractor.extractor as extractor
import Violence_Detection.infer as detector
import threading
import logging


def main(model):
    logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
    
    i3d_buffer = []
    video_queue = Queue()
    stream_queue = Queue()

    """
    Events
    """
    extract_event = threading.Event()

    """
    Start Recording
    """
    recorder.record_video_demo(frame_queue = video_queue, extract_event = extract_event, video_filename = "demo.mp4")

    i3d_frame= extractor.listener_demo(video_queue, extract_event) 
    i3d_buffer.append(i3d_frame)


    while(extract_event.isSet() == True):
        i3d_features = extractor.extract_demo(video_queue, extract_event)
        i3d_buffer.append(i3d_features)
        prediction = detector.infer(i3d_buffer, model)
        print("At: ", prediction.start)
        print("Violence: ", prediction.prediction)
        print("Violent Rate: ", prediction.score)
        i3d_buffer.pop(0)
        
if __name__ == '__main__':
    # load model
    XD_model = detector.load()
    main(XD_model)