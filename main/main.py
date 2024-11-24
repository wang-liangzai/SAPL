import os
import argparse
import time
import random
import pickle
import numpy as np
import torch.nn as nn
import torch
import json
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch import optim
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, BertTokenizer, BertConfig, BertModel
from data_utils import RGDataset, RG2Dataset
from model import DMSD, SoftEmbedding
import torch.nn.functional as F
from losses import SupConLoss2, MMD_loss, Span_loss, KL_loss
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, accuracy, macro_f1, weighted_f1
from util import set_optimizer, save_model
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(family='Times New Roman')

roberta_initialize_from_vocab = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
log_dir = 'logs/tensorboard/main'
save_folder = 'saved_models/main'
batch_size = 16
writer = SummaryWriter(log_dir)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument("--data_path", default='../data', type=str)
    parser.add_argument("--image_path", default='/home/mmn/wgj/MILNet/data/image_data', type=str)
    parser.add_argument("--save_folder", default=save_folder, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_len", default=77, type=int,
                        help="Total number of text.can not alter")
    
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch_size')
    parser.add_argument('--seed', type=int, default=512,
                        help="random seed for initialization")
    parser.add_argument("--alpha", default='0.1',
                        type=float)
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    # optimization
    parser.add_argument("--global_learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--global_pre",
                        default=1e-5, 
                        type=float)
    parser.add_argument("--global_weight_decay",
                        default=1e-5,
                        type=float)
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20,30,40,50',
                        help='where to decay lr, can be a list')
    
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # other setting
    parser.add_argument('--cosine', action='store_true', default=True,
                        help='using cosine annealing')

    opt = parser.parse_args()
    
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    return opt

def set_loader(opt):
    # construct data loader
    train_dataset = RGDataset(os.path.join(opt.data_path, 'train_id.txt'), opt.image_path, opt.max_len)  # 训练集最大长度是65
    valid_dataset = RG2Dataset(os.path.join(opt.data_path, 'valid_id.txt'), opt.image_path, opt.max_len)
    test_dataset = RG2Dataset(os.path.join(opt.data_path, 'test_id.txt'), opt.image_path, opt.max_len)
    ood_dataset = RG2Dataset(os.path.join(opt.data_path, 'ood_id.txt'), opt.image_path, opt.max_len)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                    num_workers=opt.num_workers, pin_memory=True)
    
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    
    ood_loader = DataLoader(ood_dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    return train_loader, valid_loader, test_loader, ood_loader

def set_model(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMSD()

    ce_criterion = torch.nn.CrossEntropyLoss()
    cl_criterion = SupConLoss2(temperature=opt.temp, t=0.8)
    mmd_loss_func = MMD_loss()
    kl_loss = KL_loss()

    if torch.cuda.is_available():
        model = model.cuda()
        ce_criterion = ce_criterion.cuda()
        cl_criterion = cl_criterion.cuda()
        mmd_loss_func = mmd_loss_func.cuda()
        kl_loss = kl_loss.cuda()
        cudnn.benchmark = True
    return model, ce_criterion, cl_criterion, mmd_loss_func, kl_loss

def train(train_loader, model, ce_criterion, cl_criterion, mmd_loss_func, kl_loss, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    for idx, batch in enumerate(tqdm(train_loader)):

        bert_mask, bert_mask_add, bert_indices_add, images, labels = batch  # 此处直接转为tensor，实际上bert_mask==bert_mask_add，bert_indices_add为ids

        bert_mask = torch.cat([bert_mask[0], bert_mask[1], bert_mask[2]], dim=0)  # 三种数据类型，一个是原样本，一个是反事实，一个是
        bert_mask_add = torch.cat([bert_mask_add[0], bert_mask_add[1], bert_mask_add[2]], dim=0)
        bert_indices_add = torch.cat([bert_indices_add[0], bert_indices_add[1], bert_indices_add[2]], dim=0)

        images = torch.cat([images[0], images[1], images[2]], dim=0)  # [bsz*3,3,224,224]
        labels = torch.cat([labels[0], labels[1], labels[2]], dim=0)  # [bsz*3,3,224,224]


        if torch.cuda.is_available():
            bert_mask = bert_mask.cuda(non_blocking=True)
            bert_mask_add = bert_mask_add.cuda(non_blocking=True)
            bert_indices_add = bert_indices_add.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)         # [bsz*3,1]
            # bert_mask2 = bert_mask2.cuda(non_blocking=True)
            # bert_indices_add2 = bert_indices_add2.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        output, features, mmd_feature = model(bert_mask, bert_mask_add, bert_indices_add, images, "train")   # [bsz*3,feature_dim]，output：48*2，features：48*128
        ''''''#新增加的动态路由分数
        # 这里应该是确定前1/3，然后后面的某一个跟着变动，然后相应的标签也应该换，这里可以尝试直接不换是什么效果,我要拉近的是第一部分和第二部分，因此标签=1，需要把第二部分和第三部分互换,这里标签就不用换了，因为是拉近用的
        # 应该是在train里面探索第二部分和第三部分的特征变动
        # output2, features2, mmd_feature2, pooler_output2, visual = model(bert_mask2, bert_mask2, bert_indices_add2, images, "train")
        # mmd_loss_pos = mmd_loss_func(mmd_feature[:opt.batch_size], mmd_feature[opt.batch_size:2*opt.batch_size], sample='pos')
        # mmd_loss_neg = mmd_loss_func(mmd_feature[:opt.batch_size], mmd_feature[2*opt.batch_size:3*opt.batch_size], sample='neg')
        # kl_loss_pos_1 = kl_loss(mmd_feature[:int(mmd_feature.size(0)/3)], mmd_feature[int(mmd_feature.size(0)/3):2*int(mmd_feature.size(0)/3)])
        # kl_loss_pos_2 = kl_loss(mmd_feature[int(mmd_feature.size(0) / 3):2 * int(mmd_feature.size(0) / 3)], mmd_feature[:int(mmd_feature.size(0) / 3)])
        kl_loss_neg_1 = kl_loss(mmd_feature[int(mmd_feature.size(0) / 3):2 * int(mmd_feature.size(0) / 3)], mmd_feature[2 * int(mmd_feature.size(0) / 3):3 * int(mmd_feature.size(0) / 3)])
        kl_loss_neg_2 = kl_loss(mmd_feature[2 * int(mmd_feature.size(0) / 3):3 * int(mmd_feature.size(0) / 3)], mmd_feature[int(mmd_feature.size(0) / 3):2 * int(mmd_feature.size(0) / 3)])
        mmd_loss = mmd_loss_func(mmd_feature[opt.batch_size:2*opt.batch_size], mmd_feature[2 * opt.batch_size:3 * opt.batch_size],
                                      sample='neg')
        ce_loss = ce_criterion(output, labels)
        # ce_loss2 = ce_criterion(scores, labels[:int(scores.size(0))])
        cl_loss = cl_criterion(features, labels)
        #it_loss = cl_criterion(mmd_feature, labels)


        # loss = opt.alpha * ce_loss + (1 - opt.alpha) * cl_loss + opt.alpha * (torch.exp(-kl_loss_neg_1)+torch.exp(-kl_loss_neg_2))
        loss = opt.alpha * ce_loss + (1 - opt.alpha) * cl_loss + opt.alpha * (
                    kl_loss_neg_1 + kl_loss_neg_2)

        # loss = opt.alpha * ce_loss + (1 - opt.alpha) * cl_loss
        # update metric


        losses.update(loss.item(), bsz)
        output = nn.functional.softmax(output, dim=-1)
        # scores = nn.functional.softmax(scores, dim=-1)
        # output[:len(scores)] = output[:len(scores)] + torch.tensor(2)*scores
        acc1 = accuracy(output, labels)
        top1.update(acc1[0], bsz)

        # Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg, top1.avg

def eval(val_loader, model, ce_criterion, opt):
    """validation"""
    model.eval()
    # state = torch.load('/home/mmn/wgj/DMSD-CL/DMSD-CL/main/saved_models/main/pytorch_model_aver_last.bin')
    # model.load_state_dict(state)
    losses = AverageMeter()
    top1 = AverageMeter()

    y_true = []
    y_pred = []
    with torch.no_grad():
        
        for idx, batch in enumerate(tqdm(val_loader)):
            bert_mask, bert_mask_add, bert_indices_add, images, labels = batch
            tensor_sum = torch.sum(labels)

            y_true.append(labels.numpy())
            if torch.cuda.is_available():
                bert_mask = bert_mask.cuda()
                bert_mask_add = bert_mask_add.cuda()
                bert_indices_add = bert_indices_add.cuda()
                images = images.cuda()
                labels = labels.cuda()         #[bsz,1]
            bsz = labels.shape[0]

            # forward
            output = model(bert_mask, bert_mask_add, bert_indices_add, images, 'test')

            # if tensor_sum != 137:
            #     continue
            # else:
            #     non_sarcastic = np.zeros(len(labels[:137]))  # 训练集标记为0
            #     sarcastic = np.ones(len(labels[137:274]))  # 测试集标记为1
            #
            #     # 合并训练集和测试集的嵌入和标记
            #     pooled_output = pooled_output.cpu().detach().numpy()
            #     embeddings = np.concatenate((pooled_output[:137], pooled_output[137:]), axis=0)
            #     labels = np.concatenate((non_sarcastic, sarcastic), axis=0)
            #
            #     # 使用t-SNE进行降维
            #     tsne = TSNE(n_components=2, random_state=42)
            #     embeddings_tsne = tsne.fit_transform(embeddings)
            #
            #     # 根据标记绘制数据分布图
            #     plt.figure(figsize=(10, 6))
            #     # ax = fig.add_subplot(111, projection='3d')
            #     plt.scatter(embeddings_tsne[labels == 0, 0], embeddings_tsne[labels == 0, 1], label='non_sarcastic', alpha=0.9)
            #     plt.scatter(embeddings_tsne[labels == 1, 0], embeddings_tsne[labels == 1, 1], label='sarcastic', alpha=0.9)
            #     plt.legend()
            #     plt.legend(prop=font)
            #     # plt.title('BERT Embeddings Visualization')
            #     plt.xlabel('(b)', fontsize=12, fontname='Times New Roman')
            #     plt.xticks(fontproperties=font)  # 设置 x 轴刻度的字体
            #     plt.yticks(fontproperties=font)  # 设置 y 轴刻度的字体
            #     plt.gca().xaxis.set_label_coords(0.5, -0.1)  # 调整 x 轴标签位置
            #     # plt.ylabel('t-SNE Component 2')
            #     # plt.savefig('../DP-SCAN/savemodel/dep-after.pdf')  # 将图形保存为图像文件
            #     plt.show()

            loss = ce_criterion(output, labels)

            output = nn.functional.softmax(output, dim=-1)
            # scores = nn.functional.softmax(scores, dim=-1)
            # output = output + torch.tensor(2)*scores
            y_pred.append(output.to('cpu').numpy())
            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels)
            top1.update(acc1[0], bsz)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        precision, recall, F_score = macro_f1(y_true, y_pred)
        w_pre, w_rec, w_f1 = weighted_f1(y_true, y_pred)
            
    return losses.avg, top1.avg, precision, recall, F_score, w_pre, w_rec, w_f1

def main():

    best_valid_acc = 0
    best_test_acc = 0
    best_ood_acc = 0
    opt = parse_option()
    print(opt)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    random.seed(opt.seed)
    # build data loader
    train_loader, valid_loader, test_loader, ood_loader = set_loader(opt)

    # build model and criterion
    model, ce_criterion, cl_criterion, mmd_loss_func, kl_loss = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        loss, acc = train(train_loader, model, ce_criterion, cl_criterion, mmd_loss_func, kl_loss, optimizer, epoch, opt)
        time2 = time.time()
        writer.add_scalar('train_loss', loss, global_step=epoch)
        writer.add_scalar('train_acc', acc, global_step=epoch)
        print(f'Train epoch {epoch}, total time {time2 - time1}, train_loss:{loss}, train_accuracy:{acc}')
        
        loss, val_acc, precision, recall, F_score, w_pre, w_rec, w_f1 = eval(valid_loader, model, ce_criterion, opt)
        print(f'Train epoch {epoch}, valid_loss:{loss}, valid_accuracy:{val_acc} \nprecision:{precision}, recall:{recall},F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_valid0" + ".bin"))
            print("better model")
       # eval for one epoch
        loss, test_acc, precision, recall, F_score, w_pre, w_rec, w_f1 = eval(test_loader, model, ce_criterion, opt)
        writer.add_scalar('test_loss', loss, global_step=epoch)
        writer.add_scalar('test_acc', test_acc, global_step=epoch)
        writer.add_scalar('F_score', F_score, global_step=epoch)
        writer.add_scalar('w_f1', w_f1, global_step=epoch)
        print(f'Train epoch {epoch}, test_loss:{loss}, test_accuracy:{test_acc} \nprecision:{precision}, recall:{recall}, F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_test0"  + ".bin"))
            print("better model")
        loss, ood_acc, precision, recall, F_score, w_pre,w_rec,w_f1 = eval(ood_loader, model, ce_criterion, opt)
        writer.add_scalar('OOD_loss', loss, global_step=epoch)
        writer.add_scalar('OOD_acc', ood_acc, global_step=epoch)
        print(f'Train epoch {epoch}, OOD_loss:{loss}, OOD_accuracy:{ood_acc} \nprecision:{precision}, recall:{recall},F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')
        if ood_acc > best_ood_acc:
            best_ood_acc = ood_acc
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_OOD0"  + ".bin"))
            print("better model")
            
    torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_last" + ".bin"))
    print('best accuracy: {:.2f}'.format(best_test_acc))
    print('best accuracy: {:.2f}'.format(best_ood_acc))
      

if __name__ == "__main__":
    main()



