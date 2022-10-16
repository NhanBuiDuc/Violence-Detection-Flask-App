import AVrecordeR.recorder as recorder
import Feature_Extractor.extractor as extractor
import Violence_Detection.infer as detector
def main():
    # load model
    model = detector.load()
    # 1. record
    avi_filename, wav_filenanme = recorder.record()
    # 2. record return rgb array and .wav file
    i3d_features, vggish_features = extractor.extract(avi_filename, wav_filenanme)
    
    confident_score_off, confident_score_on = detector.infer(i3d_features, vggish_features, model)
    print(confident_score_off, confident_score_on)
if __name__ == '__main__':
    main()