import numpy as np
import glob
import utils
import time
import glob

class Dataset():
    def __init__(self, args):
        self.env = args.env_name
        self.num_subgoals = args.num_subgoals
        self.state_size = args.state_size
        self.path_to_trajectories = sorted(glob.glob(args.path_to_trajectories))
        self.batch_size = args.batch_size
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.n_train = int(0.9*len(self.path_to_trajectories))
        
        self.trainidx = np.array(range(0, self.n_train))
        self.testidx = np.array(range(self.n_train, len(self.path_to_trajectories)))

        self.labels = []
        if args.pretrained_ckpt is not None:
            self.labels = np.load('./labels/' + args.pretrained_ckpt + '.npy')
        np.save('./labels/' + args.model_name + '.npy', self.labels)

    def load_feature(self, idx):
        tmp = np.load(self.path_to_trajectories[idx], encoding='bytes')
        return np.array(tmp).astype('float32')

    def load_data(self, is_training=True):
        if is_training==True:
            idx = np.random.choice(len(self.trainidx), size=self.batch_size)
            trajectories = [self.load_feature(self.trainidx[i]) for i in idx]
            if not len(self.labels):
                labels = [utils.get_labels(len(trajectories[i]), self.num_subgoals) for i in range(len(idx))] # Equi-partition subgoals
            else:
                labels = [self.labels[self.trainidx[i]] for i in idx] # Estimated subgoals
            return trajectories, labels

        else:
            feat = []
            feat.append(self.load_feature(self.testidx[self.currenttestidx]))
            if self.currenttestidx == len(self.testidx)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1

            return feat, None, done

