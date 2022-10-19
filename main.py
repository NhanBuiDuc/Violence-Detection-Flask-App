import AVrecordeR.recorder as recorder
import Feature_Extractor.extractor as extractor
import Violence_Detection.infer as detector
import threading
import time
def main(model):
    i3d_buffer = []
    vggish_buffer = []
    while(True):
        # 1. record
        avi_filename, wav_filenanme = recorder.record()
        # 2. record return rgb array and .wav file
        i3d_features, vggish_features = extractor.extract(avi_filename, wav_filenanme)

        i3d_buffer.append(i3d_features)
        vggish_buffer.append(vggish_features)
        if(len(i3d_buffer) == 5):

            i3d_features_list = i3d_buffer
            vggish_features_list = vggish_buffer
            i3d_buffer = []
            vggish_buffer = []
            confident_score = detector.infer(i3d_features_list, vggish_features_list, model)
            print("Violent score: " , confident_score)
            # Makes sure the threads have finished
            while threading.active_count() > 1: 
                time.sleep(1)
            break

if __name__ == '__main__':
    # load model
    XD_model = detector.load()
    main(XD_model)