import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
bias = False

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1  or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if bias:
            m.bias.data.fill_(0)

class ConvModel(nn.Module):

    def __init__(self, num_inputs, num_subgoals, use_rnn):

        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4, bias=bias)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, bias=bias)
        self.fc = nn.Linear(32*9*9, 256, bias=bias)
        self.lstm = nn.LSTMCell(256, 256) if use_rnn else None
        self.subgoal_linear = nn.Linear(256, num_subgoals, bias=bias)
	self.use_rnn = use_rnn

        self.apply(weights_init)

    def forward(self, inputs):

        inputs, (hx, cx) = inputs

        subgoal_values = torch.zeros(0)
        features = torch.zeros(0)

        if not self.use_rnn:
            return self.one_pass(inputs, hx, cx)
                 
        for i in range(inputs.shape[0]):
            s, x, (hx, cx) = self.one_pass(inputs[i:i+1,:,:,:], hx, cx)
            subgoal_values = torch.cat([subgoal_values, s], dim=0)
            features = torch.cat([features, x], dim=0)

        return subgoal_values, features, (hx, cx)

    def one_pass(self, x, hx, cx):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.view(x.data.cpu().numpy().shape[0], -1)))

        if self.use_rnn:
            if len(hx) == 0:
                hx = Variable(torch.zeros(1,256)).to(torch.device("cuda"))
            if len(cx) == 0:
                cx = Variable(torch.zeros(1,256)).to(torch.device("cuda"))
            hx, cx = self.lstm(x, (hx, cx))
            x = hx

        return self.subgoal_linear(x), x, (hx, cx)


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class MLPModel(nn.Module):
    def __init__(self, num_inputs, num_subgoals, use_rnn):

        super(MLPModel, self).__init__()

        self.fc1 = nn.Linear(num_inputs, 200, bias=bias)
        self.fc2 = nn.Linear(200, 200, bias=bias)
        self.fc3 = nn.Linear(200, 128, bias=bias)
        self.lstm = nn.LSTMCell(128, 128) if use_rnn else None
        self.subgoal_linear = nn.Linear(128, num_subgoals, bias=bias)

        # weight initialization
        self.apply(weights_init_mlp)
        relu_gain = nn.init.calculate_gain('relu')
        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)
        self.fc3.weight.data.mul_(relu_gain)

        if use_rnn:
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

        self.use_rnn = use_rnn

    def forward(self, inputs):

        inputs, (hx, cx) = inputs

        subgoal_values = torch.zeros(0)
        features = torch.zeros(0)

        if not self.use_rnn:
            return self.one_pass(inputs, hx, cx)

        for i in range(inputs.shape[0]):
            s, x, (hx, cx) = self.one_pass(inputs[i:i + 1, :], hx, cx) 
            subgoal_values = torch.cat([subgoal_values, s], dim=0)
            features = torch.cat([features, x], dim=0)

        return subgoal_values, features, (hx, cx)

    def one_pass(self, x, hx, cx):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        if self.use_rnn:

            if len(hx) == 0:
                hx = Variable(torch.zeros(1, 128)).to(torch.device("cuda"))
            if len(cx) == 0:
                cx = Variable(torch.zeros(1, 128)).to(torch.device("cuda"))

            hx, cx = self.lstm(x, (hx, cx))
            x = hx

        return self.subgoal_linear(x), x, (hx, cx)


