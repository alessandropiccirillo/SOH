"""
MIT数据集的结果分析

English:
    This file is used to analyze the results of the MIT dataset.
"""
import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

class Results:
    def __init__(self, root='../results/OSF results/'):
        self.root = root
        self.experiments = os.listdir(root)
        self.log_dir = None
        self.pred_label = None
        self.true_label = None
        self._update_experiments(1)

    def _update_experiments(self, e):
        experiment = 'Experiment' + str(e)
        self.log_dir = os.path.join(self.root, experiment, 'logging.txt')
        self.pred_label = os.path.join(self.root, experiment, 'pred_label.npy')
        self.true_label = os.path.join(self.root, experiment, 'true_label.npy')

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

        # 解析超参数，logging等级为CRITICAL
        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].strip()
                if ':' in params:
                    k, v = params.split(':', 1)
                    data_dict[k.strip()] = v.strip()

        # 解析train/valid/test过程中的loss
        train_data_loss = []
        train_PDE_loss = []
        train_phy_loss = []
        train_total_loss = []
        valid_data_loss = []
        test_mse = []
        test_epoch = []

        for i, line in enumerate(lines):
            # Gestione dei casi per il training
            if '[train] epoch:1 iter:1 data' in line or '[Train]' in line:
                if all(key in line for key in ['data loss:', 'PDE loss:', 'physics loss:', 'total loss:']):
                    try:
                        data_loss = float(line.split('data loss:')[1].split(',')[0])
                        PDE_loss = float(line.split('PDE loss:')[1].split(',')[0])
                        phy_loss = float(line.split('physics loss:')[1].split(',')[0])
                        total_loss = float(line.split('total loss:')[1].split('\n')[0])
                        train_data_loss.append(data_loss)
                        train_PDE_loss.append(PDE_loss)
                        train_phy_loss.append(phy_loss)
                        train_total_loss.append(total_loss)
                    except Exception as e:
                        print("Errore nel parsing della riga (train):", line, "->", e)
                else:
                    print("Stringhe di loss mancanti nella riga (train):", line)
            # Gestione dei casi per la validazione
            elif '[Valid]' in line:
                if 'MSE:' in line:
                    try:
                        valid_loss = float(line.split('MSE:')[1].split('\n')[0])
                        valid_data_loss.append(valid_loss)
                    except Exception as e:
                        print("Errore nel parsing della riga (Valid):", line, "->", e)
                else:
                    print("Stringa 'MSE:' mancante nella riga (Valid):", line)
            # Gestione dei casi per il test
            elif '[Test]' in line:
                if 'MSE:' in line:
                    try:
                        test_loss = float(line.split('MSE:')[1].split(',')[0])
                        test_mse.append(test_loss)
                        # Il test_epoch viene ricavato dalla riga precedente
                        if i > 0 and 'epoch:' in lines[i - 1]:
                            epoch_val = int(lines[i - 1].split('epoch:')[1].split(',')[0])
                            test_epoch.append(epoch_val)
                        else:
                            print("Stringa 'epoch:' mancante nella riga precedente al [Test] per la riga:", line)
                    except Exception as e:
                        print("Errore nel parsing della riga (Test):", line, "->", e)
                else:
                    print("Stringa 'MSE:' mancante nella riga (Test):", line)

        data_dict['train_data_loss'] = train_data_loss
        data_dict['train_PDE_loss'] = train_PDE_loss
        data_dict['train_phy_loss'] = train_phy_loss
        data_dict['train_total_loss'] = train_total_loss
        data_dict['valid_data_loss'] = valid_data_loss
        data_dict['test_mse'] = test_mse
        data_dict['test_epoch'] = test_epoch

        # 解析数据路径
        if len(lines) > 1:
            line1 = lines[1]
            if '.csv' in line1:
                line_temp = line1[1:-2]
                line_list = line_temp.replace('data/MIT data/', '').replace('.csv', '').replace('\'', '').split(', ')
                data_dict['IDs_1'] = line_list

        if len(lines) > 3:
            line2 = lines[3]
            if '.csv' in line2:
                line_temp = line2[1:-2]
                line_list = line_temp.replace('data/MIT data/', '').replace('.csv', '').replace('\'', '').split(', ')
                for i in range(len(line_list)):
                    line_list[i] = line_list[i].split('\\')[-1]
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
        # Modifica qui per gestire il caso in cui eval_metrix restituisca 4 valori anziché 5
        metrics = eval_metrix(pred_label, true_label)
        if len(metrics) == 4:
            MAE, MAPE, MSE, RMSE = metrics
            R2 = np.nan
        elif len(metrics) == 5:
            MAE, MAPE, MSE, RMSE, R2 = metrics
        else:
            raise ValueError("eval_metrix ha restituito un numero inatteso di metriche: {}".format(len(metrics)))

        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        MSE_list = []
        RMSE_list = []
        R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff > 0.05)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))

        start = 0
        end = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            # Calcolo le metriche per ciascun sottoinsieme
            metrics_subset = eval_metrix(pred_i, true_i)
            if len(metrics_subset) == 4:
                MAE_i, MAPE_i, MSE_i, RMSE_i = metrics_subset
                R2_i = np.nan
            elif len(metrics_subset) == 5:
                MAE_i, MAPE_i, MSE_i, RMSE_i, R2_i = metrics_subset
            else:
                raise ValueError("eval_metrix ha restituito un numero inatteso di metriche per il subset: {}".format(len(metrics_subset)))

            start = end + 1

            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(MAE_i)
            MAPE_list.append(MAPE_i)
            MSE_list.append(MSE_i)
            RMSE_list.append(RMSE_i)
            R2_list.append(R2_i)
        results_dict = {}
        results_dict['pred_label'] = pred_label_list
        results_dict['true_label'] = true_label_list
        results_dict['MAE'] = MAE_list
        results_dict['MAPE'] = MAPE_list
        results_dict['RMSE'] = RMSE_list
        results_dict['R2'] = R2_list
        return results_dict

    def get_test_results(self, e):
        '''
        解析训练和测试数据中的电池id
        English:
            Parse the battery id in the training and test sets
        :param e: experiment id
        :return:
        '''
        self._update_experiments(e)
        log_dict = self.parser_log()
        results_dict = self.parser_label()
        results_dict['channel'] = log_dict.get('IDs_2', [])
        return results_dict

    def get_battery_average(self):
        '''
        计算每次实验中所有电池的平均值
        English:
            Calculate the average value of all batteries in each experiment
        :return: dataframe，包含所有电池的平均值，每一行表示一个实验
        '''
        df_mean_values = []
        for e in range(1, 11):
            res = self.get_test_results(e)
            df_i = pd.DataFrame(res)
            df_i = df_i[['MAE', 'MAPE', 'RMSE', 'R2']]
            df_i_mean = df_i.mean(axis=0)
            df_mean_values.append(df_i_mean.values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'RMSE', 'R2'])
        df_mean.insert(0, 'experiment', range(1, 11))
        print(df_mean)
        return df_mean

    def get_experiment_average(self):
        '''
        分别获取每个测试电池在所有实验中的平均值
        English:
            Get the average value of each test battery in all experiments
        :return: dataframe，每一行是一个电池在10次实验中的平均值
        '''
        df_value_list = []
        for i in range(1, 11):
            res = self.get_test_results(i)
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
    root = '../results/OSF results/'
    writer = pd.ExcelWriter('../results/OSF_results.xlsx')
    results = Results(root)
    df_mean1 = results.get_battery_average()
    df_mean2 = results.get_experiment_average()
    df_mean1.to_excel(writer, sheet_name='battery_mean_0', index=False)
    df_mean2.to_excel(writer, sheet_name='experiment_mean_0', index=False)
    writer.save()
    print(df_mean2.mean())
