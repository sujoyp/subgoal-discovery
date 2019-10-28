import torch
import numpy as np
from model import SubgoalPrediction
import glob
import pylab as pl
import options
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import utils


def process(args):
	#args = options.parser.parse_args()
	print(args.model_name)
	device = torch.device("cuda")

	model = SubgoalPrediction(3, args.num_subgoals, args.use_rnn, useModel=False).cuda()
	model.load_state_dict(torch.load('./ckpt/' + args.model_name + '.pkl'))

	files = sorted(glob.glob('../expert_trajectories/*.npy')); files = files[int(0.9*len(files)):]
	gtlabels = utils.getGTlabels(files)

	files = sorted(glob.glob('../expert_trajectories_1/*.npy')); files = files[int(0.9*len(files)):]
	predictions = [F.softmax(model((Variable(torch.from_numpy(np.load(t, encoding='bytes')).float().cuda()), ((), ())))[0], dim=1).data.cpu().numpy() for t in files]
	predicted_labels = [np.argmax(t, axis=1) for t in predictions]
	dtw_predicted_labels = [np.argmax(utils.getAssignment(t, args.num_subgoals), axis=1) for t in predictions]

	equipartition_labels = [np.argmax(utils.get_labels(len(f), args.num_subgoals), axis=1) for f in gtlabels]

	acc1, acc2, acc3, acc4, n = 0., 0., 0., 0., 0.
	for i in range(len(gtlabels)):
	    assert gtlabels[i].shape ==equipartition_labels[i].shape
	    assert gtlabels[i].shape ==predicted_labels[i].shape
	    assert gtlabels[i].shape ==dtw_predicted_labels[i].shape
	    assert equipartition_labels[i].shape ==predicted_labels[i].shape

	    acc1 += np.sum(gtlabels[i]==equipartition_labels[i])
	    acc2 += np.sum(gtlabels[i]==predicted_labels[i])
	    acc3 += np.sum(gtlabels[i]==dtw_predicted_labels[i])
	    acc4 += np.sum(equipartition_labels[i]==predicted_labels[i])
	    n += len(gtlabels[i])

	print('Accuracy %f' %(acc1/n))
	print('Accuracy %f' %(acc2/n))
	print('Accuracy %f' %(acc3/n))
	print('Accuracy %f' %(acc4/n))
	itr = np.load('./iter_num/' + args.model_name + '.npy')
	fid = open(args.model_name + 'results.txt', 'a+')
	fid.write('%d %.2f %.2f %.2f %.2f\n' %(itr, acc1/n, acc2/n, acc3/n, acc4/n))
	fid.close()
