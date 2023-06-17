import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F



def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r = input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight), r * torch.tanh(self.bias))
        output = cosine
        return output

# ***********
# Our modules
# ***********


class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0).cuda()

    def forward(self, x):
        if self.training:
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x

class DFN(nn.Module):
  def __init__(self, input_size, hidden_size, use_bottleneck=True, bottleneck_dim=100, radius=10.0, class_num=1000):
    super(DFN, self).__init__()

    # For SEED and SEED-IV data
    input_size = input_size
    hidden_size = hidden_size

    # ++ FEATURE EXTRACTOR ++
    #fc1.weight.data.normal_(0, 0.01)
    self.feature_extractor = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.BatchNorm1d(hidden_size, momentum=0.1, affine=False),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Flatten(),
        GaussianNoise(1.0)
    )

    # set
    self.use_bottleneck = use_bottleneck

    self.bottleneck_fc = nn.Linear(hidden_size, bottleneck_dim)
    self.bottleneck = nn.Sequential(
        self.bottleneck_fc,
        nn.BatchNorm1d(bottleneck_dim, momentum=0.1, affine=False),
        nn.ReLU(),
        nn.Dropout(p=0.5)
    )

    # layer for SLR layer
    self.fc = SLR_layer(bottleneck_dim, class_num)

    # init weights for bottleneck layer
    self.bottleneck_fc.apply(init_weights)
    self.__in_features = bottleneck_dim


    self.radius = radius


  def forward(self, x):

      # flatten
      x = x.view(x.size(0), -1)

      # apply feature extractor
      x = self.feature_extractor(x)

      # apply bottleneck
      x = self.bottleneck(x)

      # apply norm
      x = self.radius * x / (torch.norm(x, dim=1, keepdim=True) + 1e-10)

      # SLR layer
      y = self.fc(x)

      return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
      # return the parameters of the deep neural network
      parameter_list = [{"params": self.feature_extractor.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                        {"params": self.bottleneck.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                        {"params": self.fc.parameters(), "lr_mult": 1, 'decay_mult': 2}]

      return parameter_list



class DiscriminatorDANN(nn.Module):
    def __init__(self, in_feature, hidden_size, radius=10.0, max_iter=10000):
        super(DiscriminatorDANN, self).__init__()

        self.radius = radius
        self.ad_layer1 = nn.Linear(in_feature, hidden_size + 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)

        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(hidden_size + 1, hidden_size + 1)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)


        self.ad_layer3 = nn.Linear(hidden_size + 1, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)

        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3, nn.Sigmoid())



    def forward(self, x, y=None):
        f2 = self.fc1(x)
        f = self.fc2_3(f2)

        return f

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]

