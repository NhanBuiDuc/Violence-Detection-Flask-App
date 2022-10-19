import threading
import torch
import numpy as np
from Violence_Detection.model import Model
from Violence_Detection import predict
import Violence_Detection.option as option
import time
from Queue import Queue
from torch.utils.data import TensorDataset, DataLoader


def inference_thread(i3d_features_list, vggish_features_list, model):
    
    device = torch.device("cuda")
    input_list = []
    for i3d_features, vggish_features in zip(i3d_features_list, vggish_features_list):
        if i3d_features.shape[0] == vggish_features.shape[0]:
            feature = np.concatenate((i3d_features, vggish_features),axis=1)
        else:# because the frames of flow is one less than that of rgb
                feature = np.concatenate((i3d_features[:-1], vggish_features), axis=1)
        input_list.append(feature)
    #convert np array to tensor
    input = torch.tensor(input_list, dtype=torch.float32)
    data_set = TensorDataset(input)
    data_loader = DataLoader(dataset=data_set, batch_size=5)
   
    print(model)
    #gt = np.load(args.gt)
    st = time.time()
    pr_auc, pr_auc_online = predict.predict(model, data_loader, device)
    print('Time:{}'.format(time.time()-st))
    # print('offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
    prediction = 0
    for i in range(len(pr_auc)):
        prediction += pr_auc_online[i]
    
    queue.enque((prediction / len(pr_auc)))

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

    confident_score = queue.dequeue()
    return confident_score

def load():
    load_thread = threading.Thread(target=load_model)
    load_thread.start()
    load_thread.join()
    
    model = queue.dequeue()
    return model