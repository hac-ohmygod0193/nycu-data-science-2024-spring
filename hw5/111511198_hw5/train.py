import os
import argparse
import torch
from torch.backends import cudnn
from config import config, dataset_config, merge_cfg_arg
from tqdm import tqdm
from dataloder import get_loader, get_loader_output
from solver_makeup import Solver_makeupGAN
import net
from torchvision.utils import save_image
from utils import to_var, de_norm
from torchvision import transforms
import warnings
# Disable all warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN')
    # general
    parser.add_argument('--data_path', default='/home/aiml/crg/data-science/lab5/submit/mtdataset/', type=str, help='training and test data path')
    parser.add_argument('--dataset', default='MAKEUP', type=str)
    parser.add_argument('--gpus', default='0', type=str, help='GPU device to train with')
    parser.add_argument('--batch_size', default='1', type=int, help='batch_size')
    parser.add_argument('--vis_step', default='260', type=int, help='steps between visualization')
    parser.add_argument('--snapshot_epoch', default='10', type=int, help='epochs between saving checkpoint')
    parser.add_argument('--task_name', default='v4', type=str, help='task name')
    parser.add_argument('--checkpoint', default='', type=str, help='checkpoint to load')
    parser.add_argument('--ndis', default='1', type=int, help='train discriminator steps')
    parser.add_argument('--G_LR', default="1e-4", type=float, help='Generator Learning rate')
    parser.add_argument('--D_LR', default="3e-4", type=float, help='Discriminator Learning rate')
    parser.add_argument('--decay', default='0', type=int, help='epochs number for training')
    parser.add_argument('--model', default='makeupGAN', type=str, help='which model to use: cycleGAN/ makeupGAN')
    parser.add_argument('--epochs', default='100', type=int, help='nums of epochs')
    parser.add_argument('--whichG', default='branch', type=str, help='which Generator to choose, normal/branch, branch means two input branches')
    parser.add_argument('--norm', default='SN', type=str, help='normalization of discriminator, SN means spectrum normalization, none means no normalization')
    parser.add_argument('--d_repeat', default='3', type=int, help='the repeat Res-block in discriminator')
    parser.add_argument('--g_repeat', default='6', type=int, help='the repeat Res-block in Generator')
    parser.add_argument('--lambda_cls', default='1', type=float, help='the lambda_cls weight')
    parser.add_argument('--lambda_rec', default='10', type=int, help='lambda_A and lambda_B')
    parser.add_argument('--lambda_his', default='1', type=float, help='histogram loss on lips')
    parser.add_argument('--lambda_skin_1', default='0.1', type=float, help='histogram loss on skin equals to lambda_his* lambda_skin')
    parser.add_argument('--lambda_skin_2', default='0.1', type=float, help='histogram loss on skin equals to lambda_his* lambda_skin')
    parser.add_argument('--lambda_eye', default='1', type=float, help='histogram loss on eyes equals to lambda_his*lambda_eye')
    parser.add_argument('--content_layer', default='r41', type=str, help='vgg layer we use to output features')
    parser.add_argument('--lambda_vgg', default='5e-3', type=float, help='the param of vgg loss')
    parser.add_argument('--cls_list', default='N,M', type=str, help='the classes of makeup to train')
    parser.add_argument('--direct', type=bool, default=True, help='direct means to add local cosmetic loss at the first, unified training')
    parser.add_argument('--lips', type=bool, default=True, help='whether to finetune lips color')
    parser.add_argument('--skin', type=bool, default=True, help='whether to finetune foundation color')
    parser.add_argument('--eye', type=bool, default=True, help='whether to finetune eye shadow color')
    args = parser.parse_args()
    return args

def train_net():
    # enable cudnn
    cudnn.benchmark = True

    data_loaders = get_loader(config, mode="train")    # return train&test
    #get the solver
    if args.model =='makeupGAN':
        solver = Solver_makeupGAN(data_loaders, config, dataset_config)
    else:
        print("model that not support")
        exit()
    solver.train()
    
def generate_img():
    dataloader_A, dataloader_B = get_loader_output()    # return train&test
    checkpoint = '_v3/resume_checkpoint'
    #checkpoint = '_/100_1109' # '_v3/80_1109'
    print(checkpoint)
    G = net.Generator_branch(config.g_conv_dim, config.g_repeat_num)
    G.load_state_dict(torch.load(os.path.join(
            config.snapshot_path, '{}_G.pth'.format(checkpoint))))
    if torch.cuda.is_available():
        G.cuda()
    output_dir = '/home/aiml/crg/data-science/lab5/submit/predictions/'
    resize_output_dir = '/home/aiml/crg/data-science/lab5/submit/resize_predictions/'
    #output_dir = './predictions/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(resize_output_dir):
        os.mkdir(resize_output_dir)
    resize_transform = transforms.Resize((128,128))
    for i, (img_A, img_B) in enumerate(tqdm((zip(dataloader_A, dataloader_B)))):
        org_A = to_var(img_A, requires_grad=False)
        ref_B = to_var(img_B, requires_grad=False)
        fake_A, fake_B = G(org_A, ref_B)
        makeup = os.path.join(output_dir, f'pred_{i}.png')
        resize_makeup = os.path.join(resize_output_dir, f'pred_{i}.png')
        save_image(de_norm(fake_A), makeup,normalize=True)
        save_image(resize_transform(de_norm(fake_A)), resize_makeup,normalize=True)
if __name__ == '__main__':
    args = parse_args()
    config = merge_cfg_arg(config, args)

    dataset_config.name = args.dataset
    print('           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰')
    print('🎵 hhey, arguments are here if you need to check 🎵')
    for arg in vars(config):
        print('{:>15}: {:>30}'.format(str(arg), str(getattr(config, arg))))
    print()
    # Create the directories if not exist
    if not os.path.exists(config.data_path):
        print("No datapath!!", config.data_path)
        exit()

    train_net()
    #generate_img()