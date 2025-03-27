'''
TJU数据集的结果分析

English:
    This file is used to analyze the results of the TJU dataset.
'''
import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

from sklearn.metrics import r2_score  # Import per calcolare R2

class Results:
    def __init__(self, root='../results/TJU results/', gap=0.07):
        self.root = root
        self.experiments = os.listdir(root)
        self.gap = gap
        self.log_dir = None
        self.pred_label = None
        self.true_label = None
        self._update_experiments(1)

    def _update_experiments(self, train_batch=0, test_batch=1, experiment=1):
        subfolder = f'{train_batch}-{test_batch}/Experiment{experiment}'
        self.log_dir = os.path.join(self.root, subfolder, 'logging.txt')
        self.pred_label = os.path.join(self.root, subfolder, 'pred_label.npy')
        self.true_label = os.path.join(self.root, subfolder, 'true_label.npy')

    def parser_log(self):
        '''
        解析train过程中产生的log文件，获取里面的数据
        English:
            Parse the log file generated during the training process to obtain the data
        :return: dict
        '''
        data_dict = {}

        with open(self.log_dir, 'r') as f:
            lines = f.readlines()

        # Parsing dei messaggi CRITICAL
        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].split('\n')[0]
                if ':' in params:
                    k, v = params.split(':', 1)
                    data_dict[k.strip()] = v.strip()

        train_data_loss = []
        train_PDE_loss = []
        train_phy_loss = []
        train_total_loss = []
        valid_data_loss = []

        test_mse = []
        test_epoch = []

        for i in range(len(lines)):
            line = lines[i]
            # Gestione righe [train] con formato specifico
            if '[train] epoch:1 iter:1 data' in line and 'data loss:' in line:
                try:
                    train_data_loss.append(
                        float(line.split('data loss:')[1].split(',')[0])
                    )
                    train_PDE_loss.append(
                        float(line.split('PDE loss:')[1].split(',')[0])
                    )
                    train_phy_loss.append(
                        float(line.split('physics loss:')[1].split(',')[0])
                    )
                    train_total_loss.append(
                        float(line.split('total loss:')[1].split('\n')[0])
                    )
                except Exception as e:
                    print("Errore nel parsing della riga [train] iter:1:", line)
            # Gestione righe [Train]
            elif '[Train]' in line and 'data loss:' in line:
                try:
                    train_data_loss.append(
                        float(line.split('data loss:')[1].split(',')[0])
                    )
                    train_PDE_loss.append(
                        float(line.split('PDE loss:')[1].split(',')[0])
                    )
                    train_phy_loss.append(
                        float(line.split('physics loss:')[1].split(',')[0])
                    )
                    train_total_loss.append(
                        float(line.split('total loss:')[1].split('\n')[0])
                    )
                except Exception as e:
                    print("Errore nel parsing della riga [Train]:", line)
            # Gestione righe [Valid]
            elif '[Valid]' in line and 'MSE:' in line:
                try:
                    valid_data_loss.append(
                        float(line.split('MSE:')[1].split('\n')[0])
                    )
                except Exception as e:
                    print("Errore nel parsing della riga [Valid]:", line)
            # Gestione righe [Test]
            elif '[Test]' in line and 'MSE:' in line:
                try:
                    test_mse.append(
                        float(line.split('MSE:')[1].split(',')[0])
                    )
                    # Si assume che la riga precedente contenga l'info sull'epoch
                    if 'epoch:' in lines[i - 1]:
                        test_epoch.append(
                            int(lines[i - 1].split('epoch:')[1].split(',')[0])
                        )
                    else:
                        print("Impossibile trovare 'epoch:' nella riga precedente a:", line)
                except Exception as e:
                    print("Errore nel parsing della riga [Test]:", line)

        data_dict['train_data_loss'] = train_data_loss
        data_dict['train_PDE_loss'] = train_PDE_loss
        data_dict['train_phy_loss'] = train_phy_loss
        data_dict['train_total_loss'] = train_total_loss
        data_dict['valid_data_loss'] = valid_data_loss
        data_dict['test_mse'] = test_mse
        data_dict['test_epoch'] = test_epoch

        # Parsing degli ID (se presenti)
        if len(lines) > 1:
            line1 = lines[1]
            if '.csv' in line1:
                line = line1[1:-2]
                line_list = line.replace('data/TJU data/', '').replace('.csv', '').replace('\'', '').split(', ')
                data_dict['IDs_1'] = line_list

        if len(lines) > 3:
            line2 = lines[3]
            if '.csv' in line2:
                line = line2[1:-2]
                line_list = line.replace('data/TJU data/', '').replace('.csv', '').replace('\'', '').split(', ')
                for j in range(len(line_list)):
                    line_list[j] = line_list[j].split('\\')[-1]
                data_dict['IDs_2'] = line_list

        return data_dict

    def parser_label(self):
        '''
        解析预测结果
        English:
            Parse the prediction results
        :return:
        '''
        pred_label = np.load(self.pred_label).reshape(-1)
        true_label = np.load(self.true_label).reshape(-1)
        # Eval_metrix ora restituisce 4 valori; calcoliamo R2 separatamente
        [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
        R2 = r2_score(true_label, pred_label)

        # plt.figure(figsize=(6, 3),dpi=200)
        # plt.plot(true_label, label='true label')
        # plt.plot(pred_label, label='pred label')
        # plt.legend()
        # plt.show()

        # 用来保存每个电池的预测结果
        # To save the prediction results of each battery
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        MSE_list = []
        RMSE_list = []
        R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff > self.gap)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))

        start = 0
        end = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            # Calcolo delle metriche per ciascuna batteria
            mae_i, mape_i, mse_i, rmse_i = eval_metrix(pred_i, true_i)
            r2_i = r2_score(true_i, pred_i)
            print('battery {} MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}, R2:{:.4f}'.format(
                i + 1, mae_i, mape_i, mse_i, rmse_i, r2_i))
            start = end + 1

            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(mae_i)
            MAPE_list.append(mape_i)
            MSE_list.append(mse_i)
            RMSE_list.append(rmse_i)
            R2_list.append(r2_i)
        # print('Mean  MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}, R2:{:.4f}'.format(MAE, MAPE, MSE, RMSE, R2))
        results_dict = {}
        results_dict['pred_label'] = pred_label_list
        results_dict['true_label'] = true_label_list
        results_dict['MAE'] = MAE_list
        results_dict['MAPE'] = MAPE_list
        # results_dict['MSE'] = MSE_list  # MSE non viene utilizzato nel dataframe finale
        results_dict['RMSE'] = RMSE_list
        results_dict['R2'] = R2_list
        return results_dict

    def get_test_results(self, train=0, test=1, e=1):
        '''
        解析训练和测试数据中的电池id
        English:
            Parse the battery id in the training and test sets
        :param e: experiment id
        :return:
        '''
        self._update_experiments(train_batch=train, test_batch=test, experiment=e)
        log_dict = self.parser_log()
        results_dict = self.parser_label()
        results_dict['channel'] = log_dict.get('IDs_2', [])
        return results_dict

    def get_battery_average(self, train_batch=0, test_batch=0):
        '''
        计算每次实验中所有电池的平均值
        English:
            Calculate the average value of all batteries in each experiment
        :param train_batch:
        :param test_batch:
        :return: dataframe，每一行是一个实验中所有电池的平均值
        '''
        df_mean_values = []
        for i in range(1, 11):
            res = self.get_test_results(train_batch, test_batch, i)
            df_i = pd.DataFrame(res)
            df_i = df_i[['MAE', 'MAPE', 'RMSE', 'R2']]
            df_mean_values.append(df_i.mean(axis=0).values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'RMSE', 'R2'])
        df_mean.insert(0, 'experiment', range(1, 11))
        print(df_mean)
        return df_mean

    def get_experiments_mean(self, train_batch=0, test_batch=0):
        '''
        分别获取每个测试电池在所有实验中的平均值
        English:
            Get the average value of each test battery in all experiments
        :return: dataframe，每一行是一个电池在10次实验中的平均值
        '''
        df_value_list = []
        for i in range(1, 11):
            res = self.get_test_results(train_batch, test_batch, i)
            df = pd.DataFrame(res)
            df = df[['channel', 'MAE', 'MAPE', 'RMSE', 'R2']]
            df = df.sort_values(by='channel')
            df.reset_index(drop=True, inplace=True)
            df_value_list.append(df[['MAE', 'MAPE', 'RMSE', 'R2']].values)
        channel = df['channel']
        columns = ['MAE', 'MAPE', 'RMSE', 'R2']

        np_array = np.array(df_value_list)
        np_mean = np.mean(np_array, axis=0)
        df_mean = pd.DataFrame(np_mean, columns=columns)
        df_mean.insert(0, column='channel', value=channel)
        print(df_mean)
        return df_mean


if __name__ == '__main__':
    root = '../results of reviewer/TJU results/'
    writer = pd.ExcelWriter('../results of reviewer/TJU_results.xlsx')
    batch = 0
    results = Results(root, gap=0.07)
    for batch in [0, 1, 2]:
        df_battery_mean = results.get_battery_average(train_batch=batch, test_batch=batch)
        df_experiment_mean = results.get_experiments_mean(test_batch=batch, train_batch=batch)
        df_battery_mean.to_excel(writer, sheet_name='battery_mean_{}'.format(batch), index=False)
        df_experiment_mean.to_excel(writer, sheet_name='experiment_mean_{}'.format(batch), index=False)
    writer.save()
    print(df_experiment_mean.mean())
