import threading
import torch
torch.cuda.empty_cache()
from Violence_Detection.model import Model
from Violence_Detection import predict
import Violence_Detection.option as option
import time
from Queue import Queue
from torch.utils.data import TensorDataset, DataLoader
from prediction import Prediction

def inference_thread(i3d_features_list, model):
    i3d_features_dict = []
    for feature_frame in i3d_features_list:
        i3d_features_dict.append(feature_frame.i3d) 
    device = torch.device("cuda")
    input = torch.tensor(i3d_features_dict, dtype=torch.float32)
    data_set = TensorDataset(input)
    data_loader = DataLoader(dataset=data_set, batch_size=5)
    #gt = np.load(args.gt)
    st = time.time()
    pr_auc, pr_auc_online = predict.predict(model, data_loader, device)
    print('Time:{}'.format(time.time()-st))
    # print('offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
    score = 0
    for i in range(len(pr_auc)):
        score += pr_auc_online[i]

    # The average of 5 prediction snipets
    average_score = ((score / len(pr_auc_online)))
    prediction = Prediction(thresh_hold=0.5)
    prediction.start = i3d_features_list[0].start
    prediction.end = i3d_features_list[-1].end
    prediction.score = average_score
    if( prediction.score >= prediction.thresh_hold):
        prediction.prediction = True
    else:
        prediction.prediction = False
    queue.enque(prediction)

def load_model():
    global queue
    queue = Queue()
    args = option.parser.parse_args()
    device = torch.device("cuda")
    model = Model(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('Violence_Detection/ckpt/rgb_xd.pkl').items()})
    print(model_dict)
    queue.enque(model)
    return model


"""
Start Infer Thread
"""
def infer(i3d_features, model):
    test_thread = threading.Thread(target=inference_thread(i3d_features, model,))
    test_thread.start()
    
    # Get return value from Thread using Queue
    confident_score = queue.dequeue()
    return confident_score

def inference(frames, model, batch, prediction):
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    i3d_features_dict = []
    for feature_frame in frames:
        i3d_features_dict.append(feature_frame.i3d) 

    input = torch.tensor(i3d_features_dict, dtype=torch.float32)
    data_set = TensorDataset(input)
    data_loader = DataLoader(dataset=data_set, batch_size=batch)
    #gt = np.load(args.gt)
    st = time.time()
    pr_auc, pr_auc_online = predict.predict(model, data_loader, device)
    print('Time:{}'.format(time.time()-st))
    # print('offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
    score = 0
    for i in range(len(pr_auc)):
        score += pr_auc_online[i]

    # The average of 5 prediction snipets
    average_score = ((score / len(pr_auc_online)))
    prediction.start = frames[0].start
    prediction.end = frames[-1].end
    prediction.score = average_score
    if( prediction.score >= prediction.thresh_hold):
        prediction.prediction = True
    else:
        prediction.prediction = False

"""
Starts load model thread
"""
def load():
    load_thread = threading.Thread(target=load_model)
    load_thread.start()
    load_thread.join()
    
    model = queue.dequeue()
    return model
