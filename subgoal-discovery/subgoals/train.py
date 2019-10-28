import torch
import torch.nn.functional as F
import torch.optim as optim
from load_data import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
import time
import pylab as pl
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def MILL(subgoal_prediction, batch_size, labels, device):
    ''' subgoal_prediction should be list of dimension (B, ) where each element is (n_steps, n_subgoals),
        labels should have same structure as subgoal_predictions but 0/1 inputs'''

    milloss = 0.
    for i in range(batch_size):
        milloss = milloss - torch.mean(torch.sum(Variable(labels[i]) \
            * torch.log(F.softmax(subgoal_prediction[i], dim=1)), dim=1))
    return milloss / batch_size
    

def CASL(x, subgoal_prediction, labels, batch_size, device):
    ''' x is list of dimension (B,) where each element is of dimension (n_steps, n_feature), 
        subgoal_prediction should be list of dimension (B, ) where each element is (n_steps, n_subgoals)
        labels should have same structure as subgoal_predictions but 0/1 inputs '''

    sim_loss = 0.
    n_tmp = batch_size * (batch_size - 1) / 2 * labels[0].shape[1]
    for i in range(batch_size):
        for j in range(i, batch_size):
            atn1 = F.softmax(subgoal_prediction[i], dim=0)
            atn2 = F.softmax(subgoal_prediction[j], dim=0)

            n1 = torch.FloatTensor([x[i].shape[0]]).cuda()
            n2 = torch.FloatTensor([x[j].shape[0]]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)
            Hf2 = torch.mm(torch.transpose(x[j], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[j], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))

            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda()))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda()))
    return sim_loss / n_tmp

def OnCL(final_features, c):
    oncl = 0.
    for feat in final_features:
        oncl = oncl + torch.sum((feat - c).pow(2))
    return oncl/len(feat)

def train(itr, dataset, args, model, optimizer, logger, device, c):

    features, labels = dataset.load_data()

    outputs = [model((Variable(torch.from_numpy(feat).float().cuda()), ((), ()))) for feat in features] 
    subgoal_predictions = [out[0] for out in outputs]
    final_features = [feat[1] for feat in  outputs]

    labels = [torch.from_numpy(lab).float().cuda() for lab in labels]
        
    milloss = MILL(subgoal_predictions, args.batch_size, labels, device)
    casloss = CASL(final_features, subgoal_predictions, labels, args.batch_size, device)
    onecloss = OnCL(final_features, c)

    total_loss = 0.5 * milloss + 0.5 * casloss + 0.01 * onecloss
        
    logger.log_value('milloss', milloss, itr)
    logger.log_value('casloss', casloss, itr)
    logger.log_value('total_loss', total_loss, itr)
    
    if itr % 1000 == 0:
        print('Iter: %d, Loss: %.2f' %(itr, total_loss))

    optimizer.zero_grad()
    total_loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

