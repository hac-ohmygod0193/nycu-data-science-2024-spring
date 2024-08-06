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
    parser.add_argument('--data_path', default='./submit/mtdataset/', type=str, help='training and test data path')
    parser.add_argument('--checkpoint', default='200_1109', type=str, help='checkpoint to load')
    parser.add_argument('--prediction_path', default='./submit/predictions', type=str, help='predictions to save')
    args = parser.parse_args()
    return args

def train_net(args):
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
    
def generate_img(args):
    dataloader_A, dataloader_B = get_loader_output(args)    # return train&test
    #checkpoint = '_/100_1109' # '_v3/80_1109'
    print(args.checkpoint)
    G = net.Generator_branch(config.g_conv_dim, config.g_repeat_num)
    G.load_state_dict(torch.load(os.path.join(
            'checkpoint', '{}_G.pth'.format(args.checkpoint))))
    if torch.cuda.is_available():
        G.cuda()
    resize_output_dir = args.prediction_path
    if not os.path.exists(resize_output_dir):
        os.mkdir(resize_output_dir)

    resize_transform = transforms.Resize((128,128))
    for i, (img_A, img_B) in enumerate(tqdm((zip(dataloader_A, dataloader_B)))):
        org_A = to_var(img_A, requires_grad=False)
        ref_B = to_var(img_B, requires_grad=False)
        fake_A, fake_B = G(org_A, ref_B)
        #makeup = os.path.join(output_dir, f'pred_{i}.png')
        resize_makeup = os.path.join(resize_output_dir, f'pred_{i}.png')
        #save_image(de_norm(fake_A), makeup,normalize=True)
        save_image(resize_transform(de_norm(fake_A)), resize_makeup,normalize=True)
if __name__ == '__main__':
    args = parse_args()
    print('           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰')
    print('🎵 hhey, arguments are here if you need to check 🎵')
    for arg in vars(args):
        print('{:>15}: {:>30}'.format(str(arg), str(getattr(args, arg))))
    print()
    # Create the directories if not exist
    if not os.path.exists(args.data_path):
        print("No datapath!!", args.data_path)
        exit()

    generate_img(args)