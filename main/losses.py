import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class SupConLoss2(nn.Module):
    def __init__(self, temperature=0.07, eps=0.5, t=1.00):
        super(SupConLoss2, self).__init__()
        self.temperature = temperature
        self.eps = torch.tensor(eps)
        self.t = t

    def forward(self, features, labels=None, mask=None):    # feature来的时候是48*128 label是48
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]  # 48
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)  # 特征展平 48*1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # 48*1和1*48，将标签相同的置为true，其它的置为false，特就是同类数据拉近，不同数据拉远
        else:
            mask = mask.float().to(device)

        anchor_feature = features

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, anchor_feature.T),  # 48*128，128*48， 每个位置的值是每一行和每一列的乘积
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 取最大的一个，每一行元素和整个转置矩阵最相关的48*1
        logits = anchor_dot_contrast - logits_max.detach()  # 减去最大值，提高数值稳定性

        similarity = torch.matmul(anchor_feature, anchor_feature.T)
        # 自适应权重分配
        mask_same_label = mask - torch.eye(batch_size).to(device)     # without self  留下的是相同类型的标签是1
        mask_diff_label = 1 - mask   # 不同类型的都再置为1
        # 相同的标签越不相似权重越高
        for i in range(batch_size):
            sample = similarity[i]
            index1 = mask_same_label[i] != 0
            sample[index1] = 1.0 - torch.softmax(sample[index1], 0)  # sample[index1]只留下有用的值，对值标签值相同的操作
            index2 = mask_diff_label[i] != 0
            sample[index2] = 1.0 + torch.softmax(sample[index2], 0)
        w_same = similarity*mask_same_label / self.t    # 上面操作的结果最终会反应到相似度矩里面
        w_diff = similarity*mask_diff_label / self.t
        logits_mask = w_same + w_diff
    
        
        mask = mask_same_label
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask   # 概率分布值乘以相应的权重
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = - (log_prob*w_same).sum(1) / mask.sum(1)

        # loss
        loss = mean_log_prob_pos.mean()
        return loss


import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance_square = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance_square) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # /len(kernel_val)

    def forward(self, source, target, sample, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        source = source.view(size=(-1, 128))
        target = target.view(size=(-1, 128))

        source_num = int(source.size()[0])

        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                       fix_sigma=fix_sigma)
        XX = torch.mean(kernels[:source_num, :source_num])
        YY = torch.mean(kernels[source_num:, source_num:])
        XY = torch.mean(kernels[:source_num, source_num:])
        YX = torch.mean(kernels[source_num:, :source_num])

        loss = XX + YY - XY - YX

        return loss
        # source_mask = source_mask.view(size=(-1,))
        # print("source ", source.shape)
        # print("source ", source_mask.shape)
        # source = source[source_mask.bool()]
        # print("source", source.shape)
    # target_mask = target_mask.view(size=(-1,))
    # print("target ", target.shape)
    # print("target ", target_mask.shape)
    # target = target[target_mask.bool()]
    # print("target", target.shape) target_num = int(target.size()[0])

class Span_loss(nn.Module):
    def __init__(self, num_label, class_weight=None):
        super().__init__()
        self.num_label = num_label
        if class_weight != None:
            self.class_weight = torch.FloatTensor(class_weight)

            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weight)  # reduction='mean'
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()  # reduction='mean'

    def forward(self, span_logits, span_label, span_mask):
        '''
        span_logits.size()==(bsz,max_length,max_length,num_labels)
        span_label.size()==span_mask.size()==(bsz,max_length,max_length)
        '''
        # print(span_mask.size(),'spanmask.size()')
        mask_pad = span_mask.view(-1) == 1
        span_label = span_label.view(size=(-1,))[mask_pad]  # (bsz*max_length*max_length,)
        span_logits = span_logits.view(size=(-1, self.num_label))[mask_pad]  # (bsz*max_length*max_length,num_labels)
        span_loss = self.loss_func(input=span_logits, target=span_label)  # (bsz*max_length*max_length,)

        # print("span_logits : ",span_logits.size())
        # print("span_label : ",span_label.size())
        # print("span_mask : ",span_mask.size())
        # print("span_loss : ",span_loss.size())

        # start_extend = span_mask.unsqueeze(2).expand(-1, -1, seq_len)
        # end_extend = span_mask.unsqueeze(1).expand(-1, seq_len, -1)
        # span_mask = span_mask.view(size=(-1,))#(bsz*max_length*max_length,)
        # span_loss *=span_mask
        # avg_se_loss = torch.sum(span_loss) / span_mask.size()[0]
        # avg_se_loss = torch.sum(span_loss) / torch.sum(span_mask).item()
        # # avg_se_loss = torch.sum(sum_loss) / bsz
        # return avg_se_loss
        return span_loss

class KL_loss(nn.Module):

    def forward(self, p, q):
        """
        Compute KL divergence between two distributions p and q.
        p, q: PyTorch tensors of shape (batch_size, seq_length, hidden_size)
        """
        # 将 P_tensor 和 Q_tensor 转换为概率分布并计算 KL 散度
        P_probs = F.softmax(p, dim=1)  # 对 hidden_size 维度进行 softmax
        Q_probs = F.softmax(q, dim=1)  # 对 hidden_size 维度进行 softmax

        # 计算 KL 散度
        kl_divergence = torch.mean(torch.sum(P_probs * (torch.log(P_probs) - torch.log(Q_probs)), dim=1))

        return kl_divergence



if __name__ == '__main__':
    import random
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    loss = SupConLoss2(t=0.8)
    # loss = SupConLoss(eps=0.0)
    features = torch.rand(12, 8)
    labels = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
    l = loss(features, labels)
    print(l)
    
