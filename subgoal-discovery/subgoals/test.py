import torch
import torch.nn.functional as F
import torch.optim as optim
from load_data import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
import scipy.io as sio
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def test(itr, dataset, args, model, logger, device):
    
    done = False
    correct = []

    while not done:
        if dataset.currenttestidx % 100 ==0:
            print('Testing test data point %d of %d' %(dataset.currenttestidx, len(dataset.testidx)))

        features, labels, done = dataset.load_data(is_training=False)

        with torch.no_grad():
            predictions = []
            for feat in features:
                s = model((Variable(torch.from_numpy(feat).float().cuda()), ((), ())))[0]
                print(s)
                tmp, _ = torch.topk(s, k=args.num_subgoals, dim=0)
                tmp = torch.sigmoid(torch.mean(tmp, 0)[0])
                predictions.append(tmp.cpu().data.numpy() > 0.5)

        predictions = np.array(predictions)
        if not len(correct):
           correct = predictions==labels
        else:
           correct = np.concatenate([correct, predictions==labels], axis=0)

    accuracy = np.mean(correct)
    print('Classification accuracy %f' %accuracy)
        
    logger.log_value('Test Classification Accuracy', accuracy, itr)
