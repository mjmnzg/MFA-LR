import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import manifold

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def loss_adv(features, ad_net, logits=None):

    ad_out = ad_net(features, logits)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def cosine_matrix(x,y):
    x=F.normalize(x,dim=1)
    y=F.normalize(y,dim=1)
    xty=torch.sum(x.unsqueeze(1)*y.unsqueeze(0),2)
    return 1-xty

def SM(Xs, Xt, Ys, Yt, Cs_memory, Ct_memory, Wt=None, decay=0.3):
    # Clone memory
    Cs = Cs_memory.clone()
    Ct = Ct_memory.clone()

    r = torch.norm(Xs, dim=1)[0]
    Ct = r*Ct / (torch.norm(Ct, dim=1, keepdim=True)+1e-10)
    Cs = r*Cs / (torch.norm(Cs, dim=1, keepdim=True)+1e-10)

    K = Cs.size(0)
    # for each class
    for k in range(K):
        Xs_k = Xs[Ys==k]
        Xt_k = Xt[Yt==k]

        if len(Xs_k)==0:
            Cs_k = 0.0
        else:
            Cs_k = torch.mean(Xs_k,dim=0)

        if len(Xt_k) == 0:
            Ct_k = 0.0
        else:
            if Wt is None:
                Ct_k = torch.mean(Xt_k,dim=0)
            else:
                Wt_k = Wt[Yt==k]
                Ct_k = torch.sum(Wt_k.view(-1, 1) * Xt_k, dim=0) / (torch.sum(Wt_k) + 1e-5)

        Cs[k, :] = (1-decay) * Cs_memory[k, :] + decay * Cs_k
        Ct[k, :] = (1-decay) * Ct_memory[k, :] + decay * Ct_k

    Dist = cosine_matrix(Cs, Ct)

    return torch.sum(torch.diag(Dist)), Cs, Ct

def robust_pseudo_loss(output,label,weight,q=1.0):
    weight[weight<0.5] = 0.0
    one_hot_label=torch.zeros(output.size()).scatter_(1,label.cpu().view(-1,1),1).cuda()
    mask=torch.eq(one_hot_label,1.0)
    output=F.softmax(output,dim=1)
    mae=(1.0-torch.masked_select(output,mask)**q)/q
    return torch.sum(weight*mae)/(torch.sum(weight)+1e-10)

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)