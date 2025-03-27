'''
MIT数据集的结果分析

English:
    This file is used to analyze the results of the MIT dataset.
'''
import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

class Results:
    def __init__(self, root='../results of reviewer/MIT results/'):
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
                params = line.split('\t')[-1].split('\n')[0]
                if ':' in params:
                    k, v = params.split(':', 1)
                    data_dict[k] = v

        # 解析train/valid/test过程中的loss
        train_data_loss = []
        train_PDE_loss = []
        train_phy_loss = []
        train_total_loss = []
        valid_data_loss = []

        test_mse = []
        test_epoch = []

        for i, line in enumerate(lines):
            if '[train] epoch:1 iter:1 data' in line or '[Train]' in line:
                # Verifica che la riga contenga tutti i token necessari
                if all(token in line for token in ['data loss:', 'PDE loss:', 'physics loss:', 'total loss:']):
                    try:
                        train_data_loss.append(float(line.split('data loss:')[1].split(',')[0]))
                        train_PDE_loss.append(float(line.split('PDE loss:')[1].split(',')[0]))
                        train_phy_loss.append(float(line.split('physics loss:')[1].split(',')[0]))
                        train_total_loss.append(float(line.split('total loss:')[1].split('\n')[0]))
                    except Exception as e:
                        print("Errore nel parsing della riga (train):", line, e)
                else:
                    print("Riga train saltata, formato non conforme:", line)
            elif '[Valid]' in line:
                if 'MSE:' in line:
                    try:
                        valid_data_loss.append(float(line.split('MSE:')[1].split('\n')[0]))
                    except Exception as e:
                        print("Errore nel parsing della riga (valid):", line, e)
            elif '[Test]' in line:
                if 'MSE:' in line:
                    try:
                        test_mse.append(float(line.split('MSE:')[1].split(',')[0]))
                        # Assumiamo che la riga precedente contenga 'epoch:'
                        test_epoch.append(int(lines[i - 1].split('epoch:')[1].split(',')[0]))
                    except Exception as e:
                        print("Errore nel parsing della riga (test):", line, e)

        data_dict['train_data_loss'] = train_data_loss
        data_dict['train_PDE_loss'] = train_PDE_loss
        data_dict['train_phy_loss'] = train_phy_loss
        data_dict['train_total_loss'] = train_total_loss
        data_dict['valid_data_loss'] = valid_data_loss
        data_dict['test_mse'] = test_mse
        data_dict['test_epoch'] = test_epoch

        # 解析数据路径
        line1 = lines[1]
        if '.csv' in line1:
            line = line1[1:-2]
            line_list = line.replace('data/MIT data/', '').replace('.csv','').replace('\'','').split(', ')
            data_dict['IDs_1'] = line_list

        line2 = lines[3]
        if '.csv' in line2:
            line = line2[1:-2]
            line_list = line.replace('data/MIT data/', '').replace('.csv', '').replace('\'', '').split(', ')
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
        # Modifica: unpacking per 4 valori anziché 5
        [MAE, MAPE, RMSE, R2] = eval_metrix(pred_label, true_label)
        # Se necessario, è possibile stampare le metriche globali:
        # print('Mean  MAE:{:.4f}, MAPE:{:.4f}, RMSE:{:.4f}, R2:{:.4f}'.format(MAE, MAPE, RMSE, R2))

        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        RMSE_list = []
        R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff > 0.05)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))

        start = 0
        for end in local_minima:
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            # Modifica: unpacking per 4 valori
            [MAE_i, MAPE_i, RMSE_i, R2_i] = eval_metrix(pred_i, true_i)
            start = end + 1

            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(MAE_i)
            MAPE_list.append(MAPE_i)
            RMSE_list.append(RMSE_i)
            R2_list.append(R2_i)
            
        results_dict = {
            'pred_label': pred_label_list,
            'true_label': true_label_list,
            'MAE': MAE_list,
            'MAPE': MAPE_list,
            'RMSE': RMSE_list,
            'R2': R2_list
        }
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
        :return: dataframe, dove ogni riga rappresenta un esperimento
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
        分别获取每个测试电池在 tutti gli esperimenti
        English:
            Get the average value of each test battery in all experiments
        :return: dataframe, dove ogni riga rappresenta la media di un test battery su 10 esperimenti
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
    # Utilizzo della cartella "results of reviewer" per gli esperimenti e salvataggio del file Excel
    root = '../results of reviewer/MIT results/'
    writer = pd.ExcelWriter('../results of reviewer/MIT_results.xlsx')
    results = Results(root)
    df_mean1 = results.get_battery_average()
    df_mean2 = results.get_experiment_average()

    df_mean1.to_excel(writer, sheet_name='battery_mean_0', index=False)
    df_mean2.to_excel(writer, sheet_name='experiment_mean_0', index=False)
    writer.save()
    print(df_mean2.mean())
