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

def gen_new_labels(dataset, model, args, device):
    
    labels = []
    for i, t in enumerate(dataset.path_to_trajectories):
        trajectory = np.load(t)
        output = model((Variable(torch.from_numpy(trajectory).float().cuda()), ((), ())))[0]
        lab = utils.getAssignment(F.softmax(output, dim=1).data.cpu().numpy(), args.num_subgoals)
        lab = np.argmax(F.softmax(output, dim=1).data.cpu().numpy(), axis=1)
        lab = np.array([np.eye(output.shape[1])[l] for l in lab])
        labels.append(lab)
   
    dataset.labels = labels
    np.save('./labels/' + args.model_name + '.npy', labels)


        

    
