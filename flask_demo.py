import numpy as np
from flask import Flask, request, render_template, json, jsonify
from flask import Response
from StreamingInstance import StreamingInstance 
import multiprocessing
import threading
from imutils.video import VideoStream
from multiprocessing import Manager

app = Flask(__name__)

instances = {}

@app.route('/initialize_camera')
def initialize_camera():
	args = 	request.args
	id = args.get('id')
	id = 1
	stream = StreamingInstance(id = id, video_filename="demo2.mp4")
	instances.update( {str(stream.id):stream} )
	stream.start()
	response = app.response_class(
        response=json.dumps("TRUE"),
        status=200,
        mimetype='application/json'
    )
	return response

@app.route('/get_camera')
def get_camera():
	return render_template('index.html')


@app.route("/xd")
def xd():
	global instances

	args = 	request.args
	id = args.get('id')
	id = 1

	stream = instances.get(str(id))
	prediction = stream.get_prediction()
	if(prediction.prediction != None):
		return jsonify(
				start = str(prediction.start_datetime()),
				end = str(prediction.end_datetime()),
				score = str(prediction.score),
				prediction = str(prediction.prediction),
				thresh_hold = prediction.thresh_hold
		)
	else:
		response = app.response_class(
        response=json.dumps("NO PREDICTION"),
        status=200,
        mimetype='application/json'
    )
	return response
	
@app.route("/video_feed")
def video_feed():
	global instances

	args = 	request.args
	id = args.get('id')
	id = 1


	stream = instances.get(str(id))
	return Response(stream.raw_stream(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/end_instance")
def end_instance():
	global instances

	args = 	request.args
	id = args.get('id')
	id = 1

	stream = instances.get(str(id))
	stream.end()
	response = app.response_class(
        response=json.dumps("ENDED THE STREAM"),
        status=200,
        mimetype='application/json'
    )
	return response
# def main():
# 	global outputFrame, lock
# 	vs = VideoStream(src=0).start()
# 	time.sleep(2.0)

# 	net = serializing_model()


# 	# loop over the frames from the video stream
# 	while True:
# 		# Preprossesing the frames to obtain detections
# 		blob, frame = preprocess_video_frames(vs)

# 		# Obtaining detections from each frame
# 		detections = get_detections_from_frames(blob, net)
		
# 		# loop over the detections
# 		for key in np.arange(0, detections.shape[2]):
# 			# extract the confidence associated with the prediction
# 			confidence = detections[0, 0, key, 2]

# 			# filter out weak detections
# 			idx = filter_out_detections(detections, confidence, key)

# 			if(idx == None):
# 				break
# 			else:
# 				# Get the bounding boxes for the detections
# 				box = draw_bounding_box(detections, frame, key)
				
# 				# Get predictions for the boxes
# 				label = predict_class_labels(confidence, idx)

# 				# Draw predictions on frames
# 				frame = draw_predictions_on_frames(box, label, frame, idx)

# 		with lock:
# 				outputFrame = frame.copy()

# @app.route("/violence_score")
# def violence_score():
# 	return Response(main())
	
if __name__ == "__main__":

	app.run(threaded = False)
	