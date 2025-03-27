import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

class Results:
    def __init__(self, root='../results of reviewer/HUST results/'):
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
        """
        解析train过程中产生的log文件，获取里面的数据
        English:
            Parse the log file generated during the training process to obtain the data
        :return: dict
        """
        data_dict = {}

        with open(self.log_dir, 'r') as f:
            lines = f.readlines()

        # 解析超参数，logging等级为CRITICAL
        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].split('\n')[0]
                k, v = params.split(':')
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
            if ('[train] epoch:1 iter:1 data' in line) or ('[Train]' in line):
                if ('data loss:' in line and 'PDE loss:' in line and
                    'physics loss:' in line and 'total loss:' in line):
                    try:
                        data_loss_val = float(line.split('data loss:')[1].split(',')[0])
                        PDE_loss_val = float(line.split('PDE loss:')[1].split(',')[0])
                        phy_loss_val = float(line.split('physics loss:')[1].split(',')[0])
                        total_loss_val = float(line.split('total loss:')[1].split('\n')[0])
                        train_data_loss.append(data_loss_val)
                        train_PDE_loss.append(PDE_loss_val)
                        train_phy_loss.append(phy_loss_val)
                        train_total_loss.append(total_loss_val)
                    except (IndexError, ValueError) as e:
                        print("Errore nel parsing della riga di training:", line)
                else:
                    print("Riga inattesa nel training:", line)
            elif '[Valid]' in line:
                if 'MSE:' in line:
                    try:
                        mse_val = float(line.split('MSE:')[1].split('\n')[0])
                        valid_data_loss.append(mse_val)
                    except (IndexError, ValueError) as e:
                        print("Errore nel parsing della riga di validazione:", line)
                else:
                    print("Riga inattesa nella validazione:", line)
            elif '[Test]' in line:
                if 'MSE:' in line:
                    try:
                        mse_val = float(line.split('MSE:')[1].split(',')[0])
                        test_mse.append(mse_val)
                    except (IndexError, ValueError) as e:
                        print("Errore nel parsing della riga di test (MSE):", line)
                    # Cerca di ottenere l'epoch dalla riga precedente
                    if i > 0 and 'epoch:' in lines[i - 1]:
                        try:
                            epoch_val = int(lines[i - 1].split('epoch:')[1].split(',')[0])
                            test_epoch.append(epoch_val)
                        except (IndexError, ValueError) as e:
                            print("Errore nel parsing dell'epoch dalla riga precedente:", lines[i - 1])
                    else:
                        print("Riga precedente non contiene 'epoch:' per la riga di test:", line)
                else:
                    print("Riga inattesa nel test:", line)

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
            line_temp = line1[1:-2]
            line_list = line_temp.replace('data/HUST data/', '').replace('.csv', '').replace('\'', '').split(', ')
            data_dict['IDs_1'] = line_list

        line2 = lines[3]
        if '.csv' in line2:
            line_temp = line2[1:-2]
            line_list = line_temp.replace('data/HUST data/', '').replace('.csv', '').replace('\'', '').split(', ')
            data_dict['IDs_2'] = line_list

        return data_dict

    def parser_label(self):
        """
        解析预测结果
        English:
            Parse the prediction results
        :return:
        """
        pred_label = np.load(self.pred_label).reshape(-1)
        true_label = np.load(self.true_label).reshape(-1)
        # Supponiamo che eval_metrix restituisca 4 valori: MAE, MAPE, RMSE, R2
        [MAE, MAPE, RMSE, R2] = eval_metrix(pred_label, true_label)

        # Lista per salvare i risultati per ogni batteria
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        RMSE_list = []
        R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff > 0.1)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))
        start = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            try:
                [MAE_i, MAPE_i, RMSE_i, R2_i] = eval_metrix(pred_i, true_i)
            except Exception as e:
                print("Errore nel parsing dei risultati per batteria {}: {}".format(i, e))
                continue
            start = end + 1

            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(MAE_i)
            MAPE_list.append(MAPE_i)
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
        """
        解析训练和测试数据中的电池id
        English:
            Parse the battery id in the training and test data
        :param e: experiment id
        :return:
        """
        self._update_experiments(e)
        log_dict = self.parser_log()
        results_dict = self.parser_label()
        results_dict['channel'] = log_dict.get('IDs_2', [])
        return results_dict

    def get_battery_mean(self):
        df_mean_values = []
        for e in range(1, 11):
            res = self.get_test_results(e=e)
            df = pd.DataFrame(res)
            df = df[['MAE', 'MAPE', 'RMSE', 'R2']]
            df_i_mean = df.mean(axis=0)
            df_mean_values.append(df_i_mean.values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'RMSE', 'R2'])
        df_mean.insert(0, column='experiment', value=np.arange(1, 11))
        print(df_mean)
        return df_mean

    def get_experiments_mean(self):
        """
        分别获取每个测试电池在所有实验中的平均值
        English:
            Get the average value of each test battery in all experiments
        :return: dataframe，每一行 è un battery in 10 experiments
        """
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
        df_mean['channel'] = df_mean['channel'].apply(lambda x: x.replace('-', '--'))
        print(df_mean)
        return df_mean

if __name__ == '__main__':
    # Crea la directory per i risultati se non esiste
    root = '../results of reviewer/HUST results/'
    writer = pd.ExcelWriter('../results of reviewer/HUST_results.xlsx')
    results = Results(root)
    
    df_battery_mean = results.get_battery_mean()
    df_experiment_mean = results.get_experiments_mean()
    df_battery_mean.to_excel(writer, sheet_name='battery_mean_0', index=False)
    df_experiment_mean.to_excel(writer, sheet_name='experiment_mean_0', index=False)
    
    writer.save()
