import copy
from torch import nn

class MinigridNet(nn.Module):
  
    def __init__(self, input_dim, output_dim):
        super().__init__()
        h, w, c = input_dim
        
        if h != 84:
            raise ValueError(f"Expecting input height: {84}, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: {84}, got: {w}")

        self.online_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten())
        self.output_dim = output_dim
        self.online_linear = nn.Sequential(nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target_conv = copy.deepcopy(self.online_conv)
        self.target_linear = copy.deepcopy(self.online_linear)
        
        # Q_target parameters are frozen.
        for p in self.target_conv.parameters():
            p.requires_grad = False
        for p in self.target_linear.parameters():
            p.requires_grad = False
        

    def reset_linear(self,use_cuda):
        self.online_linear = nn.Sequential(nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )        
        self.target_linear = copy.deepcopy(self.online_linear)
        if use_cuda:
            self.online_linear = self.online_linear.to(device='cuda')
            self.target_linear = self.target_linear.to(device='cuda')

    def forward(self, input, model):
        conv_input = input
        if model == 'online':
            conv_res = self.online_conv(conv_input)
            linear_input = conv_res
            return self.online_linear(linear_input)
        elif model == 'target':
            conv_res = self.target_conv(conv_input)
            linear_input = conv_res
            return self.target_linear(linear_input)
