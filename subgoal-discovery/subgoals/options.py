import argparse

parser = argparse.ArgumentParser(description='subgoals')
parser.add_argument('--lr', type=float, default=0.0001,help='learning rate (default: 0.0001)')
parser.add_argument('--env-name', type=str, help='name of the environment')
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model-name', default='subgoalprediction', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--state-size', help='size of state (square matrix or square)')
parser.add_argument('--num-similar', default=3, help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=100000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--num-subgoals', type=int, default=15, help='number of subgoals (default: 4)')
parser.add_argument('--path-to-trajectories', type=str, help='path pointing to the demonstrations')
parser.add_argument('--one-class', action="store_true", help='when learning one class model')



