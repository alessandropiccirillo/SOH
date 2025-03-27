'''
解析fine-tuning的结果

English:
Parse the fine-tuning results
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

from utils.util import eval_metrix

class Results:
    def __init__(self, root='../results of reviewer finetuning/XJTU-TJU', gap=0.07):
        self.root = root
        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory non trovata: {root}")
        self.experiments = os.listdir(root)
        try:
            self.dataset = os.path.basename(os.path.dirname(root))
        except Exception as e:
            self.dataset = ""
            print("Errore nel parsing del dataset:", e)
        try:
            tmp = root.split('tuning/')
            if len(tmp) > 1:
                tmp_parts = tmp[1].split('-')
                self.source = tmp_parts[0].strip()
                self.target = tmp_parts[1].rstrip('/').strip() if len(tmp_parts) > 1 else ""
            else:
                basename = os.path.basename(root.rstrip('/'))
                if '-' in basename:
                    self.source, self.target = basename.split('-', 1)
                else:
                    self.source = self.target = ""
        except Exception as e:
            self.source = ""
            self.target = ""
            print("Errore nel parsing di source/target:", e)
        self.gap = gap
        self.log_dir = None
        self.pred_label = None
        self.true_label = None
        self._update_experiments(1)

    def _update_experiments(self, batch, experiment=1):
        # Controlla se esiste la cartella "batchX" all'interno della root
        batch_folder = os.path.join(self.root, f'batch{batch}')
        if os.path.exists(batch_folder):
            # Se esiste, la struttura è: root/batchX/ExperimentY
            subfolder = os.path.join(f'batch{batch}', f'Experiment{experiment}')
        else:
            # Altrimenti, usa direttamente ExperimentY
            subfolder = f'Experiment{experiment}'
        self.log_dir = os.path.join(self.root, subfolder, 'logging.txt')
        self.pred_label = os.path.join(self.root, subfolder, 'pred_label.npy')
        self.true_label = os.path.join(self.root, subfolder, 'true_label.npy')

    def parser_log(self):
        data_dict = {}
        with open(self.log_dir, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'CRITICAL' in line:
                try:
                    params = line.split('\t')[-1].strip()
                    k, v = params.split(':')
                    data_dict[k.strip()] = v.strip()
                except Exception as e:
                    print("Errore nel parsing dei parametri:", e)
            if 'Source only:' in line:
                source_only = {}
                try:
                    text = line.split('Source only:')[-1]
                    parts = text.split('|')
                    source_only['task'] = parts[0].strip()
                    source_only['mse'] = float(text.split('MSE:')[1].split(',')[0].strip())
                    source_only['mae'] = float(text.split('MAE:')[1].split(',')[0].strip())
                    source_only['mape'] = float(text.split('MAPE:')[1].split(',')[0].strip())
                    source_only['rmse'] = float(text.split('RMSE:')[1].split('\n')[0].strip())
                    self.source_only = source_only
                except Exception as e:
                    print("Errore nel parsing di 'Source only':", e)

        train_data_loss = []
        valid_data_loss = []
        for line in lines:
            if '[train] epoch:1 iter:1 data' in line:
                try:
                    loss = float(line.split('data loss:')[1].split(',')[0].strip())
                    train_data_loss.append(loss)
                except Exception as e:
                    print("Errore nel parsing di train_data_loss:", e)
            elif '[Train]' in line:
                try:
                    loss = float(line.split('data loss:')[1].split(',')[0].strip())
                    train_data_loss.append(loss)
                except Exception as e:
                    print("Errore nel parsing di train_data_loss:", e)
            elif '[Valid]' in line:
                try:
                    loss = float(line.split('MSE:')[1].split('\n')[0].strip())
                    valid_data_loss.append(loss)
                except Exception as e:
                    print("Errore nel parsing di valid_data_loss:", e)

        data_dict['train_data_loss'] = train_data_loss
        data_dict['valid_data_loss'] = valid_data_loss

        if len(lines) > 1 and '.csv' in lines[1]:
            line = lines[1][1:-2]
            line_list = line.replace(f"data/{self.dataset} data/", "").replace('.csv', '').replace('\'', '').split(', ')
            data_dict['IDs_1'] = line_list

        if len(lines) > 3 and '.csv' in lines[3]:
            line = lines[3][1:-2]
            line_list = line.replace(f"data/{self.dataset} data/", "").replace('.csv', '').replace('\'', '').split(', ')
            line_list = [x.split('\\')[-1] for x in line_list]
            data_dict['IDs_2'] = line_list

        return data_dict

    def parser_label(self):
        pred_label = np.load(self.pred_label).reshape(-1)
        true_label = np.load(self.true_label).reshape(-1)
        # Effettua l'unpacking di 4 valori (MAE, MAPE, RMSE, R2)
        [MAE, MAPE, RMSE, R2] = eval_metrix(pred_label, true_label)
        # Preparazione delle liste per salvare i risultati per ciascun battery
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        RMSE_list = []
        R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff > self.gap)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))

        start = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
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

    def get_test_results(self, batch=1, e=1):
        self._update_experiments(batch=batch, experiment=e)
        log_dict = self.parser_log()
        results_dict = self.parser_label()
        results_dict['channel'] = log_dict.get('IDs_2', [])
        return results_dict

    def get_experiments_mean(self, batch=0, num=10):
        df_value_list = []
        for i in range(1, num + 1):
            res = self.get_test_results(batch, i)
            df = pd.DataFrame(res)
            if 'channel' in df.columns:
                df = df[['channel', 'MAE', 'MAPE', 'RMSE', 'R2']].sort_values(by='channel').reset_index(drop=True)
            else:
                df = df[['MAE', 'MAPE', 'RMSE', 'R2']]
            df_value_list.append(df[['MAE', 'MAPE', 'RMSE', 'R2']].values)
        channel = df['channel'] if 'channel' in df.columns else None
        columns = ['MAE', 'MAPE', 'RMSE', 'R2']
        np_array = np.array(df_value_list)
        np_mean = np.mean(np_array, axis=0)
        df_mean = pd.DataFrame(np_mean, columns=columns)
        if channel is not None:
            df_mean.insert(0, column='channel', value=channel)
        print(df_mean)
        return df_mean

    def get_battery_average(self, batch, num=10):
        df_mean_values = []
        for i in range(1, num + 1):
            res = self.get_test_results(batch, i)
            df_i = pd.DataFrame(res)
            df_i = df_i[['MAE', 'MAPE', 'RMSE', 'R2']]
            df_mean_values.append(df_i.mean(axis=0).values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'RMSE', 'R2'])
        df_mean.insert(0, 'experiment', range(1, num + 1))
        print(df_mean)
        return df_mean

    def get_source_only(self):
        print(self.source_only)
        df = pd.DataFrame(self.source_only, index=[1])
        return df


if __name__ == '__main__':
    import os
    source = 'TJU'
    target = 'XJTU'
    
    com_root = os.path.join('..', 'results of reviewer finetuning', f'{source}-{target}')
    if not os.path.exists(com_root):
        raise FileNotFoundError(f"Directory non trovata: {com_root}")
    
    writer_path = os.path.join('..', 'results of reviewer finetuning', f'{source}-{target}.xlsx')
    writer = pd.ExcelWriter(writer_path)
    
    xjtu_gap = 0.05
    tju_gap = 0.07
    results = Results(com_root, gap=tju_gap)
    
    for batch in range(6):
        df_experiment_mean = results.get_experiments_mean(batch=batch, num=9)
        df_battery_mean = results.get_battery_average(batch=batch, num=9)
        df_source_only = results.get_source_only()
    
        mean = df_battery_mean.mean(axis=0)
        std = df_battery_mean.std(axis=0)
        df_battery_mean = pd.concat([df_battery_mean, pd.DataFrame([mean]), pd.DataFrame([std])], ignore_index=True)
    
        df_battery_mean.to_excel(writer, f'battery_mean_{batch}', index=False)
        df_source_only.to_excel(writer, f'source_only_{batch}', index=False)
        df_experiment_mean.to_excel(writer, f'experiment_mean_{batch}', index=False)
    
    writer.close()
