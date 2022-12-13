import numpy as np
from flask import Flask, request, render_template, json, jsonify
from flask import Response
from StreamingInstance import StreamingInstance 
import threading
from multiprocessing import Manager
import os
import sys
from waitress import serve
import requests
from flask_cors import CORS, cross_origin

app = Flask('app')
cors = CORS(app)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'
instances = {}


		
@cross_origin(supports_credentials=True)
@app.route('/start', methods = ['GET', 'POST'])
def start():
	args = 	request.args
	id = args.get('connection_string')
	id = 1
	stream = StreamingInstance(id = id, video_filename="demo.mp4", )
	instances.update( {str(stream.id):stream} )
	stream.start()
	response = app.response_class(
        response=json.dumps("TRUE"),
        status=200,
        mimetype='application/json'
    )
	return response

@cross_origin(supports_credentials=True)
@app.route('/', methods = ['GET', 'POST'])
def get_camera():
	return render_template('index.html')

@cross_origin(supports_credentials=True)
@app.route("/send_event", methods = ['GET', 'POST'])
def send_event():
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

@cross_origin(supports_credentials=True)
@app.route("/xd", methods = ['GET', 'POST'])
def xd():
	try:
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
			mimetype='application/json')
		return response
	except:
		response = app.response_class(
			response=json.dumps("NO PREDICTION"),
			status=200,
			mimetype='application/json')
		return response

@cross_origin(supports_credentials=True)
@app.route("/video_feed", methods = ['GET', 'POST'])
def video_feed():
	global instances

	args = 	request.args
	id = args.get('id')
	id = 1

	stream = instances.get(str(id))
	try:
		return Response(stream.generate(),
			mimetype = "multipart/x-mixed-replace; boundary=frame")
		# return StreamingHttpResponse(stream.generate(),content_type="multipart/x-mixed-replace;boundary=frame")
	except:
		response = app.response_class(
        response=json.dumps("Not Ready Yet"),
        status=200,
        mimetype='application/json')
		return response

@cross_origin(supports_credentials=True)
@app.route("/stop", methods = ['GET', 'POST'])
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
	host_ip = "127.0.0.1"
	port = 8000
	serve(app, host=host_ip, port=8000, threads= 10)
	# app.run(host = '0.0.0.0', port = 5000)