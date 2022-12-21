import numpy as np
from flask import Flask, request, render_template, json, jsonify
from flask import Response
from StreamingInstance import StreamingInstance 


def start():
	id = 1
	stream = StreamingInstance(id = id,)
	stream.start()
if __name__ == "__main__":
	start()