import torch.nn as nn
import torch.nn.functional as F
import torch
from Module.attention import CBAM

class DuelingDQN(nn.Module):
    def __init__(self, w, channels, outputs, device):
        super(DuelingDQN, self).__init__()

        # Number of Linear input connections depends on output of conv1d layers
        # and therefore the input image size, so compute it.
        def conv1d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1))

        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3, stride=1)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(128)
        self.device = device

        # Just convenience!!!
        conv1_out_w = conv1d_size_out(w)
        conv2_out_w = conv1d_size_out(conv1_out_w)
        conv3_out_w = conv1d_size_out(conv2_out_w)
        conv3_out = conv3_out_w * 128

        # CBAM
        self.attn1 = CBAM(conv1_out_w)
        self.attn2 = CBAM(conv2_out_w)
        self.attn3 = CBAM(conv3_out_w)

        self.advantage_vector = nn.Linear(conv3_out, outputs)
        self.state_value = nn.Linear(conv3_out, 1)

        # Called with either one element to determine next action, or a batch

    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float().to(self.device)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x) # x.shape=(1, 32, 21)
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1], 1))
        x = self.attn1(x)
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1], 1))
        x = self.attn2(x)
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1], 1))
        x = self.attn3(x)
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))

        advantage_flatten = x.view(x.size(0), -1)  # advantage
        state_value_flatten = x.view(x.size(0), -1)  # q value

        advantage = self.advantage_vector(advantage_flatten)  # advantage values #relu爛掉
        state_value = self.state_value(state_value_flatten) # state value  #relu爛掉
        advantage_average = torch.mean(advantage)

        new_state_value = state_value + (advantage - advantage_average)
        return new_state_value
