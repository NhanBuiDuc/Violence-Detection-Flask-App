

class xd_req():
    def __init__(self, start = None, end = None, score = None, prediction = None, thresh_hold = None, connection_string = None):
        self.start = start
        self.end = end
        self.score = score
        self.prediction = prediction
        self.thresh_hold = thresh_hold
        self.connection_string = connection_string
    def toJson(self):
        data = {"start": self.start[0], "end": self.end[0], "score": self.score[0], "prediction": self.prediction[0], "thresh_hold": self.thresh_hold, "connection_string": self.connection_string}
        return data