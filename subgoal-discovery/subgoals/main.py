from __future__ import print_function
import argparse, os, torch, numpy as np, time
from model import MLPModel, ConvModel
from load_data import Dataset
from test import test
from train import train
from gen_new_labels import gen_new_labels
from tensorboard_logger import Logger
import options
import torch.optim as optim
from torch.autograd import Variable

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def get_c(dataset, model, args):
    features, _ = dataset.load_data()
    outputs = [model((Variable(torch.from_numpy(feat).float().cuda()), ((), ()))) for feat in features]
    final_features = torch.zeros(0)
    for feat in outputs:
        final_features = torch.cat([final_features, feat[1]], dim=0)
        c = torch.mean(final_features, dim=0)
        np.save('./c/' + args.model_name + '.npy', c.cpu().data.numpy())
        c = Variable(c, requires_grad=False)
    return c 

if __name__ == '__main__':

    args = options.parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    if not os.path.exists('./ckpt/'):
       os.makedirs('./ckpt/')
    if not os.path.exists('./iter_num/' + args.model_name):
       os.makedirs('./iter_num/' + args.model_name)
    if not os.path.exists('./logs/' + args.model_name):
       os.makedirs('./logs/' + args.model_name)
    if not os.path.exists('./labels/' + args.model_name):
       os.makedirs('./labels/' + args.model_name)
    if not os.path.exists('./c/'):
       os.makedirs('./c/')

    dataset = Dataset(args)
    change_itr = range(8000, 100000, 4000)
    logger = Logger('./logs/' + args.model_name)
    if args.env_name == 'bimgame':
        model = ConvModel(3, args.num_subgoals, use_rnn=False).to(device)
    else:
        model = MLPModel(46, args.num_subgoals, use_rnn=False).to(device)

    start_itr = 0
    c = []
    if args.one_class:
      if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load('./ckpt/' + args.pretrained_ckpt + '.pkl'))
        start_itr = np.load('./iter_num/' + args.pretrained_ckpt + '.npy')
        c = torch.from_numpy(np.load('./c/' + args.pretrained_ckpt + '.npy')).float().to(device)
      # computing initial c for one-class out-of-set estimation
      if len(c) == 0:
        c = get_c(dataset, model, args)
       
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    
    for itr in range(start_itr, args.max_iter):
       train(itr, dataset, args, model, optimizer, logger, device, c)
       if  itr % 500 == 0:
          torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
          np.save('./iter_num/' + args.model_name + '.npy', itr)
          np.save('./labels/' + args.model_name + '.npy', dataset.labels)
       if itr in change_itr:
          gen_new_labels(dataset, model, args, device)
       if args.one_class and itr % 50 == 0 and itr <= 500:
          c = get_c(dataset, model, args)

    
