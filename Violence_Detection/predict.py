from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch

def predict(model, data_loader, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        pred2 = torch.zeros(0).to(device)
        for i, input in enumerate(data_loader):
            input = input[0].to(device)
            
            logits, logits2 = model(inputs=input, seq_len=None)
            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))
            '''
            online detection
            '''
            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            sig2 = torch.mean(sig2, 0)

            sig2 = torch.unsqueeze(sig2, 1) ##for audio
            pred2 = torch.cat((pred2, sig2))

            pred = list(pred.cpu().detach().numpy())
            pred2 = list(pred2.cpu().detach().numpy())
            
        return pred, pred2



