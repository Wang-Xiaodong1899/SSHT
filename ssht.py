import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
from math import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from loss import bnm
from loss import adbnm
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

from randaugment import RandAugmentMC

from loss import entropy, adentropy
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001,
                     power=0.75, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr #* param_lr[i]
        i += 1
    return optimizer

def linear_rampup(current):
    current = np.clip(current, 0.0, 1.0)
    return float(current)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(x, size):
    s = list(x.shape)
    ##5,3,224,224
    ###-1,5,3,224,224
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, lambda_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch)

##transform
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 3e-4
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        # transforms.RandomCrop(crop_size),
        transforms.RandomCrop(size=224,
                              padding=int(224*0.125),
                              padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

class TransformFixMatch(object):
    def __init__(self):
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def data_load(args): 
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_tar_unl = open(args.t_dset_path_unl).readlines()
    txt_test = open(args.t_dset_path_unl).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train(),root = args.root)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dsets["target_unl"] = ImageList_idx(txt_tar_unl, transform=TransformFixMatch(),root = args.root)
    dset_loaders["target_unl"] = DataLoader(dsets["target_unl"], batch_size=train_bs*args.mu, shuffle=True, num_workers=args.worker, drop_last=True)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test(),root = args.root)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netG, netB, netC, flag=False):
    netG.eval()
    netB.eval()
    netC.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netG(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netG = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netG = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # modelpath = args.output_dir_src + '/source_G.pt'   
    # netG.load_state_dict(torch.load(modelpath))
    # modelpath = args.output_dir_src + '/source_B.pt'   
    # netB.load_state_dict(torch.load(modelpath))
    # modelpath = args.output_dir_src + '/source_C.pt'    
    # netC.load_state_dict(torch.load(modelpath))
    args.modelpath = args.output_dir_src + '/'+args.net+'_'+'source_G.pt'   
    netG.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/'+args.net+'_'+'source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/'+args.net+'_'+'source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netG.eval()
    netB.eval()
    netC.eval()

    param_group = []
    for k, v in netG.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    ####add
    optimizer_f = optim.SGD(list(netC.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])


    acc_s_te, acc_list = cal_acc(dset_loaders['test'], netG, netB, netC, True)
    print('acc_:',acc_s_te)
    print('----------------------------------------------------------------\n')
    max_iter = args.max_epoch * len(dset_loaders["target_unl"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    pbar = tqdm(total=max_iter)
    while iter_num < max_iter:
        try:
            (inputs_u_w, inputs_u_s), label_u, tar_idx_ = iter_test_unl.next()
        except:
            iter_test_unl = iter(dset_loaders["target_unl"])
            (inputs_u_w, inputs_u_s), label_u, tar_idx_ = iter_test_unl.next()

        ###add
        try:
            inputs_x, targets_x, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_x, targets_x, tar_idx = iter_test.next()

        if inputs_x.size(0) == 1:
            continue
        netG.train()
        netB.train()
        netC.train()
        # MEM
        if iter_num == 0 and args.cls_par > 0:
            netG.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['test'], netG, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netG.train()
            netB.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        ###add
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, iter_num,init_lr=args.lr)

        batch_size = inputs_x.shape[0]
        inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).cuda()
        # inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).cuda()
        targets_x = targets_x.cuda()
        feat_ = netB(netG(inputs))
        feat_u_w, feat_u_s = feat_[batch_size:].chunk(2)
        logits = netC(feat_)


        # logits = de_interleave(logits, 2*args.mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold_f).float()
        # print(mask.shape)
        if iter_num == 1:
            targets_u = mem_label[tar_idx_].cuda()
        Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()
        loss = Lx+ args.lu*Lu#* linear_rampup(iter_num)
        optimizer.zero_grad()
        optimizer_f.zero_grad()
        if args.method!='CDL':
            loss.backward(retain_graph = True)
            optimizer.step()
            optimizer_f.step()
            optimizer.zero_grad()
            optimizer_f.zero_grad()
        # feat_u = netB(netG(inputs_u_w.cuda()))
        # feat_u2 = netB(netG(inputs_u_s.cuda()))
        # optimizer.zero_grad()
        # optimizer_f.zero_grad()
        if args.method == 'S+T':
            optimizer.zero_grad()
            optimizer_f.zero_grad()
        elif args.method == 'Fix':
            ###
            optimizer.zero_grad()
            optimizer_f.zero_grad()
        elif args.method == 'ENT':
            loss_t = entropy(netC, feat_u, args.lamda)
            loss_t.backward()
            optimizer_f.step()
            optimizer.step()
        elif args.method == 'MME':
            loss_t = adentropy(netC, feat_u_w, args.lamda)
            loss_t.backward()
            optimizer_f.step()
            optimizer.step()
        elif args.method == 'BNM':
            loss_t = bnm(netC, feat_u_w, args.lamda)
            loss_t.backward()
            optimizer_f.step()
            optimizer.step()
        elif args.method == 'CDL':
            loss_t = bnm(netC, feat_u_w, args.lamda)
            loss_t2 = bnm(netC,feat_u_s,args.lamda)
            loss_t = (loss_t+loss_t2)/2
            loss_all = loss+args.trade_off*loss_t
            loss_all.backward()
            # print(netB.bottleneck.weight.grad)
            # break
            optimizer_f.step()
            optimizer.step()
        else:
            raise ValueError('Method cannot be recognized.')
        pbar.update(1)
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netG.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netG, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netG, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netG.train()
            netB.train()
            netC.train()
    f.close()
    pbar.close()
    if args.issave:   
        torch.save(netG.state_dict(), osp.join(args.output_dir, "target_G_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netG, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netG, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netG(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSHT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=60, help="max iterations")
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=48, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=0.003, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet34', help="alexnet, vgg16, resnet50, res101,resnet34")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--threshold_f', type=int, default=0.80)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='origin')
    parser.add_argument('--output_src', type=str, default='origin')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--num', type=int, default=3,help='number of labeled examples in the target')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda_u', default=75, type=float)
    parser.add_argument('--T', type=float, default=0.05, metavar='T',help='temperature (default: 0.05)')

    parser.add_argument('--lamda', type=float, default=1.5, metavar='LAM',help='value of lamda')
    parser.add_argument('--method', type=str, default='CDL')

    parser.add_argument('--mu', default=2, type=int,help='coefficient of unlabeled batch size')
    parser.add_argument('--trade_off', default=1, type=float,help='trade off between Fix and BNM')
    parser.add_argument('--lu', default=2.5, type=float,help='coefficient of unlabeled loss')

    args = parser.parse_args()
    print(args.net)
    print('---------------------------------------------------------')
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        folder = './data/txt/'
        args.s_dset_path = folder + args.dset + '/' + 'labeled_source_images_'+names[args.s] + '.txt'
        args.t_dset_path = folder + args.dset + '/' + 'labeled_target_images_'+names[args.t] + '_'+str(args.num)+'.txt'
        args.t_dset_path_unl = folder + args.dset + '/' + 'unlabeled_target_images_'+names[args.t] + '_'+str(args.num)+'.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        ###add
        args.savename = args.method+'_'+args.net
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)
        # break
