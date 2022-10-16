import threading
import torch
import numpy as np
from Violence_Detection.model import Model
from Violence_Detection.dataset import Dataset
from Violence_Detection import predict
import Violence_Detection.option as option
import time
from Queue import Queue

def inference_thread(i3d_features: np.array, vggish_features: np.array, model):
    
    device = torch.device("cuda")

    if i3d_features.shape[0] == vggish_features.shape[0]:
        feature = np.concatenate((i3d_features, vggish_features),axis=1)

    else:# because the frames of flow is one less than that of rgb
            feature = np.concatenate((i3d_features[:-1], vggish_features), axis=1)

    #convert np array to tensor
    input = torch.tensor(feature, dtype=torch.float32)
    print(model)
    #gt = np.load(args.gt)
    st = time.time()
    pr_auc, pr_auc_online = predict.predict(model, input, device)
    print('Time:{}'.format(time.time()-st))
    print('offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))

def load_model():
    global queue
    queue = Queue()
    args = option.parser.parse_args()
    device = torch.device("cuda")
    model = Model(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('Violence_Detection/ckpt/wsanodet_mix2.pkl').items()})
    print(model_dict)
    queue.enque(model)

def infer(i3d_features, vggish_features, model):
    test_thread = threading.Thread(target=inference_thread(i3d_features, vggish_features, model,))
    test_thread.start()

def load():
    load_thread = threading.Thread(target=load_model)
    load_thread.start()
    load_thread.join()
    
    model = queue.dequeue()
    return model