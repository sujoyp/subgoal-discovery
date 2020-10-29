import numpy as np 
import glob
import pylab as pl
import options
from torch.autograd import Variable
import torch
import time
from model import ConvModel, MLPModel
import os
args = options.parser.parse_args()
pl.rcParams['axes.linewidth'] = 2.9

files = sorted(glob.glob(args.path_to_trajectories + '/*.npy'))
files = files[int(0.9*len(files)):]

device = torch.device('cuda')
if args.env_name == 'bimgame':
    model = ConvModel(3, args.num_subgoals, use_rnn=False).to(device)
else:
    model = MLPModel(46, args.num_subgoals, use_rnn=False).to(device)

model.load_state_dict(torch.load('./ckpt/' + args.pretrained_ckpt + '.pkl'))
color = ['b', 'g', 'r', 'm', 'y', 'k', 'b', 'g', 'r', 'm', 'y', 'k', 'b', 'g', 'r', 'm', 'y', 'k', 'b', 'g', 'r', 'm', 'y', 'k', 'b', 'g', 'r', 'm', 'y', 'k']
color = np.array(color)

for file in files:
	A = np.load(file, encoding='bytes')
	subgoals, features = model((Variable(torch.from_numpy(A).float().cuda()), ((), ())))[:2]
	subgoals = subgoals.cpu().data.numpy()
	features = features.cpu().data.numpy()
	subgoals = np.argmax(subgoals, axis=1)
	print(len(set(subgoals)))
	pl.scatter(A[:,0], A[:,1], c=color[subgoals])

pl.savefig('test.png', format='png', dpi=300)
