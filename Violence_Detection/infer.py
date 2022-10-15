from torch.utils.data import DataLoader
import torch
import numpy as np
from Violence_Detection.model import Model
from Violence_Detection.dataset import Dataset
from Violence_Detection.test import test
import Violence_Detection.option as option
import time

def infer(rgb_features_dict):
    args = option.parser.parse_args()
    device = torch.device("cuda")

    model = Model(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('Violence_Detection/ckpt/wsanodet_mix2.pkl').items()})
    print(model)
    #gt = np.load(args.gt)
    st = time.time()
    pr_auc, pr_auc_online = test(model, rgb_features_dict, device)
    print('Time:{}'.format(time.time()-st))
    print('offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
