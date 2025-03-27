import torch
import torch.nn as nn
import numpy as np
import os
import time
from utils.util import AverageMeter, get_logger
from Model.Compare_Models import MLP, CNN
from Model.Model import LR_Scheduler
from dataloader.dataloader import XJTUdata, HUSTdata, MITdata, TJUdata
import argparse

def safe_save(filename, array, retries=3, delay=1):
    """
    Prova a salvare il file. Se il file esiste già o è bloccato, lo rimuove e riprova.
    retries: numero di tentativi
    delay: attesa (in secondi) tra un tentativo e l'altro
    """
    for i in range(retries):
        try:
            if os.path.exists(filename):
                os.remove(filename)
            np.save(filename, array)
            return
        except PermissionError as e:
            time.sleep(delay)
    # Se dopo i tentativi non riesce a salvare, rilancia l'eccezione
    raise PermissionError(f"Impossibile salvare il file {filename} dopo {retries} tentativi.")

class Trainer():
    def __init__(self, model, train_loader, valid_loader, test_loader, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.save_dir = args.save_folder
        if os.path.exists(self.save_dir) and not os.path.isdir(self.save_dir):
            raise Exception(f"'{self.save_dir}' esiste ed non è una directory!")
        os.makedirs(self.save_dir, exist_ok=True)
        self.epochs = args.epochs
        self.logger = get_logger(os.path.join(self.save_dir, args.log_dir))

        self.loss_meter = AverageMeter()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.warmup_lr)
        self.scheduler = LR_Scheduler(optimizer=self.optimizer,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def train_one_epoch(self, epoch):
        self.model.train()
        self.loss_meter.reset()
        for (x1, _, y1, _) in self.train_loader:
            x1 = x1.to(self.device)
            y1 = y1.to(self.device)

            y_pred = self.model(x1)
            loss = self.loss_func(y_pred, y1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_meter.update(loss.item())
        info = '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch, self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def valid(self, epoch):
        self.model.eval()
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1, _, y1, _) in self.valid_loader:
                x1 = x1.to(self.device)
                y1 = y1.to(self.device)

                y_pred = self.model(x1)
                loss = self.loss_func(y_pred, y1)
                self.loss_meter.update(loss.item())
        info = '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch, self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def test(self):
        self.model.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1, _, y1, _) in self.test_loader:
                x1 = x1.to(self.device)
                y_pred = self.model(x1)

                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label, axis=0)
        pred_label = np.concatenate(pred_label, axis=0)
        if self.save_dir is not None:
            # Normalizzazione dei percorsi di salvataggio
            true_label_path = os.path.normpath(os.path.join(self.save_dir, 'true_label.npy'))
            pred_label_path = os.path.normpath(os.path.join(self.save_dir, 'pred_label.npy'))
            try:
                safe_save(true_label_path, true_label)
            except Exception as e:
                self.logger.error(f"Errore nel salvataggio di {true_label_path}: {e}")
                raise e
            try:
                safe_save(pred_label_path, pred_label)
            except Exception as e:
                self.logger.error(f"Errore nel salvataggio di {pred_label_path}: {e}")
                raise e
        return true_label, pred_label

    def train(self):
        min_loss = 100
        early_stop = 0
        for epoch in range(1, self.epochs + 1):
            early_stop += 1
            train_loss = self.train_one_epoch(epoch)
            current_lr = self.scheduler.step()
            valid_loss = self.valid(epoch)
            if valid_loss < min_loss and self.test_loader is not None:
                min_loss = valid_loss
                true_label, pred_label = self.test()
                early_stop = 0
            if early_stop > 10:
                break
        self.clear_logger()

def load_model(args):
    if args.model == 'MLP':
        model = MLP()
    elif args.model == 'CNN':
        model = CNN()
    return model

def load_TJU_data(args, small_sample=None):
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []

    # I file con unit digit 5 o 9 sono del test set
    mod = [(5,9), (4,8), (5,9)]
    if args.in_same_batch:
        batchs = os.listdir(root)
        batch = batchs[args.tju_batch]
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for i, f in enumerate(files):
            id = i + 1
            if id % 10 == mod[args.tju_batch][0] or id % 10 == mod[args.tju_batch][1]:
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

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for TJU dataset')
    parser.add_argument('--dataset', type=str, default='TJU', help='XJTU, HUST, MIT, TJU')
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

if __name__ == '__main__':
    args = get_args()
    setattr(args, 'model', 'CNN')  # Seleziona il modello: MLP o CNN

    if args.dataset == 'TJU':
        # Per il dataset TJU consideriamo i batch 0, 1, 2 (o come definito in args)
        tju_batch_values = [0, 1, 2]
        for i in range(len(tju_batch_values)):
            setattr(args, 'tju_batch', tju_batch_values[i])
            for e in range(10):
                # Costruisci il percorso di salvataggio evitando duplicazioni indesiderate
                setattr(args, 'save_folder', os.path.join('./results of reviewer',
                                        f'{args.dataset}-{args.model} results/{tju_batch_values[i]}-{tju_batch_values[i]}/Experiment{e+1}'))
                os.makedirs(args.save_folder, exist_ok=True)

                model = load_model(args)
                data_loader = load_TJU_data(args)
                trainer = Trainer(model, data_loader['train'], data_loader['valid'], data_loader['test'], args)
                trainer.train()
