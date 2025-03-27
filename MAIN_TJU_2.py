from dataloader.dataloader import XJTUdata, TJUdata
from Model.Model import PINN
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for TJU dataset')
    parser.add_argument('--data', type=str, default='TJU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--in_same_batch', type=bool, default=True, help='Se train e test sono nello stesso batch')
    parser.add_argument('--train_batch', type=int, default=-1, choices=[-1,0,1,2],
                        help='Se -1, legge tutti i dati e li divide in train e test; altrimenti, legge il batch specificato')
    parser.add_argument('--test_batch', type=int, default=-1, choices=[-1,0,1,2],
                        help='Se -1, legge tutti i dati e li divide in train e test; altrimenti, legge il batch specificato')
    parser.add_argument('--batch', type=int, default=0, choices=[0,1,2])
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max, z-score')

    # Parametri scheduler
    parser.add_argument('--epochs', type=int, default=200, help='numero di epoche')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='epoche di warmup')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='learning rate durante il warmup')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate base')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='learning rate finale')
    parser.add_argument('--lr_F', type=float, default=0.001, help='learning rate per F')

    # Parametri del modello
    parser.add_argument('--F_layers_num', type=int, default=3, help='numero di layer per F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='dimensione degli hidden layer di F')

    # Parametri della loss
    parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.05, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='directory per il log, se None non salva')
    # Modifica del parametro save_folder: ora i risultati verranno salvati in "results of reviewer"
    parser.add_argument('--save_folder', type=str, default='results of reviewer', help='directory di salvataggio')

    args = parser.parse_args()

    return args


def load_TJU_data(args, small_sample=None):
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []

    # I file con unit digit 5 o 9 sono del test set
    mod = [(5,9), (4,8), (5,9)]
    if args.in_same_batch:
        batchs = os.listdir(root)
        batch = batchs[args.batch]
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for i, f in enumerate(files):
            id = i + 1
            if id % 10 == mod[args.batch][0] or id % 10 == mod[args.batch][1]:
                test_list.append(os.path.join(batch_root, f))
                print(f)
            else:
                train_list.append(os.path.join(batch_root, f))
        if small_sample is not None:
            train_list = train_list[:small_sample]
        train_loader = data.read_all(specific_path_list=train_list)
        test_loader = data.read_all(specific_path_list=test_list)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    else:
        batchs = os.listdir(root)
        train_loader = data.read_one_batch(args.train_batch)
        test_loader = data.read_one_batch(args.test_batch)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    return dataloader


def main():
    args = get_args()
    batchs = [0, 1, 2]

    for batch in batchs:
        setattr(args, 'in_same_batch', True)
        setattr(args, 'batch', batch)
        for e in range(10):
            # Salvataggio dei risultati all'interno della cartella "results of reviewer"
            if args.in_same_batch:
                save_folder = os.path.join('results of reviewer', f'{batch}-{batch}', f'Experiment{e+1}')
            else:
                save_folder = os.path.join('results of reviewer', f'{args.train_batch}-{args.test_batch}', f'Experiment{e+1}')
            setattr(args, "save_folder", save_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            dataloader = load_TJU_data(args)
            pinn = PINN(args)
            pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])


def small_sample():
    args = get_args()
    num_battery = 2
    batch = 2
    for e in range(10):
        setattr(args, 'in_same_batch', True)
        setattr(args, 'batch', batch)
        # Salvataggio dei risultati nella cartella "results of reviewer (small sample ...)"
        setattr(args, 'save_folder', f'results of reviewer (small sample {num_battery})/{batch}-{batch}/Experiment{e+1}')
        setattr(args, 'batch_size', 128)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        dataloader = load_TJU_data(args, small_sample=num_battery)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])


if __name__ == '__main__':
    main()
