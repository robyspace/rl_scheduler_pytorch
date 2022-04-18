import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):
    def __init__(self, w, channels, outputs, device):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.device = device

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv1d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1))

        #
        conv1_out_w = conv1d_size_out(w)
        conv2_out_w = conv1d_size_out(conv1_out_w)
        conv3_out_w = conv1d_size_out(conv2_out_w)
        conv3_out = conv3_out_w * 128
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        self.head = nn.Linear(conv3_out, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float()
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        # x.view(x.size(2))
        return self.head(x)
