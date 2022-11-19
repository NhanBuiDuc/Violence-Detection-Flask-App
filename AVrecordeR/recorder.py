from AVrecordeR.video_recorder import VideoRecorder

# def record(frame_queue, extract_event, wanted_fps, stream_event = None, stream_queue = None):
#     global video_thread
#     # not threads but objects
#     video_thread = VideoRecorder()
#     video_thread.start(stream_queue, frame_queue, wanted_fps, extract_event, stream_event = stream_event)

def record(video_filename):
    video_thread = VideoRecorder(video_filename=video_filename)
def record_webcam_demo(frame_queue, extract_event):
    global video_thread
    # not threads but objects
    video_thread = VideoRecorder()
    video_thread.start_webcam_demo(frame_queue, extract_event)

# def record_video(frame_queue, extract_event, wanted_fps, video_filename, stream_event = None, stream_queue = None):
#     global video_thread
#     # not threads but objects
#     video_thread = VideoRecorder(video_filename)
#     video_thread.start_video(stream_queue, frame_queue, wanted_fps, extract_event, stream_event = stream_event)

def record_video_demo(frame_queue, extract_event, video_filename):
    global video_thread
    # not threads but objects
    video_thread = VideoRecorder(video_filename = video_filename)
    video_thread.start_video_demo(frame_queue, extract_event)