
from dataloader.dataloader import XJTUdata, MITdata, HUSTdata, TJUdata, OSFdata
from Model.Model import PINN
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for OSF data, V4, W5, W7, W8, W9 dataset')
    parser.add_argument('--data', type=str, default='OSF data, V4, W5, W7, W8, W9', help='XJTU, HUST, MIT, TJU, OSF')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch')
    parser.add_argument('--early_stop', type=int, default=1000, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=300, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='lr of F')

    # model related
    # parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    # parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.5, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=80, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results/OSF results', help='save folder')

    args = parser.parse_args()

    return args


def load_OSF_data(args,small_sample=None):
    test_id = ['V4', 'W9']
    data = OSFdata(root='data/OSF data',args=args)
    train_list = []
    test_list = []
    files = os.listdir('data/OSF data')
    for f in files:
        if f[:-4] in test_id:
            test_list.append(f'data/OSF data/{f}')
        else:
            train_list.append(f'data/OSF data/{f}')
    if small_sample is not None:
        train_list = train_list[:small_sample]

    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader


def main():
    args = get_args()
    for e in range(10):
        setattr(args, 'save_folder', f'results/OSF results/Experiment{e + 1}')
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        dataloader = load_OSF_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])


def small_sample():
    args = get_args()
    for n in [1,2,3,4]:
        for e in range(10):
            setattr(args,'save_folder',f'results/OSF results (small sample {n})/Experiment{e+1}')
            setattr(args,'batch_size',128)
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)
            dataloader = load_OSF_data(args,small_sample=n)
            pinn = PINN(args)
            pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])


if __name__ == '__main__':
    pass
    main()
    # small_sample()