class Prediction():
    def __init__(self, start = None, end = None, score = None, prediction = None, thresh_hold = None):
        self.start = start
        self.end = end
        self.score = score
        self.prediction = prediction
        self.thresh_hold = thresh_hold

        
    def start_datetime(self):
        return self.start.strftime("%Y-%m-%d %H:%M:%S")
    def end_datetime(self):
        # return self.end.strftime("%m/%d/%Y %H:%M:%S")
        return self.end.strftime("%Y-%m-%d %H:%M:%S")