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
	stream = StreamingInstance(id = id, video_filename="demo.mp4")
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

	
if __name__ == "__main__":

	app.run(threaded = False)
	