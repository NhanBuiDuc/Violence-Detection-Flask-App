import numpy as np
from flask import Flask, request, render_template, json, jsonify
from flask import Response
from StreamingInstance import StreamingInstance 
import threading
from multiprocessing import Manager
import os
import sys
app = Flask('app')
app_thread = threading.Thread(name="Webapp",target=app.run, args = ())
instances = {}


# def thread_webAPP():
#     app = Flask(__name__)

#     @app.route("/")
#     def nothing():
#         return "Hello World!"

#     app.run(debug=True, use_reloader=False)

@app.route('/start')
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

@app.route('/')
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
	return Response(stream.generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/stop")
def stop():
	global instances

	args = 	request.args
	id = args.get('id')
	id = 1

	stream = instances.get(str(id))
	stream.stop()
	response = app.response_class(
        response=json.dumps("STOPED THE STREAM"),
        status=200,
        mimetype='application/json'
    )
	return response

# @app.route("/quit")
# def quit():
# 	quit()

if __name__ == "__main__":

	app_thread.setDaemon = True
	app_thread.start()