import os
import re
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def extract_mape_from_log(file_path):
    """
    Estrae tutti i valori di MAPE da un file di log e li converte in percentuale.
    """
    mape_values = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                matches = re.findall(r'MAPE:\s*([0-9]+\.?[0-9]*)', line)
                for match in matches:
                    mape_value = float(match) * 100  # Conversione in percentuale
                    mape_values.append(mape_value)
    except Exception as e:
        print(f"Errore nella lettura di {file_path}: {e}")
    return mape_values

def compute_average_mape_from_experiment_logs():
    """
    Apre una finestra per selezionare un file Excel contenente i percorsi delle cartelle degli esperimenti,
    estrae i valori di MAPE dai file di log (nome fisso 'logging') e calcola la media in percentuale.
    """
    root = tk.Tk()
    root.withdraw()
    excel_file = filedialog.askopenfilename(
        title="Seleziona il file Excel con i percorsi degli esperimenti",
        filetypes=[("Excel files", "*.xlsx;*.xls")]
    )
    
    if not excel_file:
        print("Nessun file Excel selezionato per gli esperimenti. Uscita.")
        return None
    
    try:
        # Legge la prima colonna contenente i percorsi
        df = pd.read_excel(excel_file, dtype=str)
        experiment_folders = df.iloc[:, 0].dropna().tolist()
    except Exception as e:
        print(f"Errore nella lettura del file Excel: {e}")
        return None
    
    all_mape_values = []
    for exp_folder in experiment_folders:
        exp_folder = str(exp_folder).strip()
        if not os.path.isdir(exp_folder):
            print(f"Percorso non valido: {exp_folder}")
            continue
        # Si assume che il file di log si chiami esattamente "logging"
        log_file_path = os.path.join(exp_folder, "logging")
        if os.path.exists(log_file_path):
            mape_values = extract_mape_from_log(log_file_path)
            all_mape_values.extend(mape_values)
        else:
            print(f"File 'logging' non trovato in: {exp_folder}")
    
    if all_mape_values:
        average_mape = sum(all_mape_values) / len(all_mape_values)
        print("\n==============================")
        print(f"Media del MAPE (in percentuale) dai log: {average_mape:.2f}%")
        print("==============================\n")
        return average_mape
    else:
        print("Nessun valore di MAPE trovato nei log.")
        return None

def compute_sheet_mape_averages(file_path):
    """
    Calcola la media (non in percentuale) per ogni colonna dei fogli specificati
    nel file Excel.
    
    I fogli presi in considerazione sono:
    battery_mean_0, battery_mean_1, battery_mean_2, battery_mean_3, battery_mean_4, battery_mean_5
    """
    sheets = [
        "battery_mean_0", "battery_mean_1", "battery_mean_2",
        "battery_mean_3", "battery_mean_4", "battery_mean_5"
    ]
    averages = {}
    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        print(f"Errore nell'apertura del file Excel {file_path}: {e}")
        return None

    for sheet in sheets:
        if sheet not in xls.sheet_names:
            print(f"Foglio {sheet} non trovato nel file.")
            continue
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
        except Exception as e:
            print(f"Errore nella lettura del foglio {sheet}: {e}")
            continue
        
        # Calcola la media per ogni colonna con dati numerici (senza moltiplicare per 100)
        sheet_avgs = {}
        for col in df.columns:
            try:
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if not col_data.empty:
                    sheet_avgs[col] = col_data.mean()
            except Exception as e:
                print(f"Errore nel calcolo della media per la colonna {col} del foglio {sheet}: {e}")
        averages[sheet] = sheet_avgs
    
    return averages

def compute_overall_mape_percentage(file_path):
    """
    Calcola la media percentuale complessiva per il solo valore MAPE
    aggregando i dati dai fogli battery_mean_0, battery_mean_1, battery_mean_2,
    battery_mean_3, battery_mean_4, battery_mean_5.
    
    Si assume che in ciascun foglio esista una colonna denominata "MAPE" contenente valori in formato decimale.
    Il risultato viene convertito in percentuale.
    """
    sheets = [
        "battery_mean_0", "battery_mean_1", "battery_mean_2",
        "battery_mean_3", "battery_mean_4", "battery_mean_5"
    ]
    all_mape_values = []
    
    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        print(f"Errore nell'apertura del file Excel {file_path}: {e}")
        return None

    for sheet in sheets:
        if sheet not in xls.sheet_names:
            print(f"Foglio {sheet} non trovato nel file.")
            continue
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
        except Exception as e:
            print(f"Errore nella lettura del foglio {sheet}: {e}")
            continue
        
        if "MAPE" in df.columns:
            try:
                mape_data = pd.to_numeric(df["MAPE"], errors="coerce").dropna()
                all_mape_values.extend(mape_data.tolist())
            except Exception as e:
                print(f"Errore nell'elaborazione della colonna MAPE nel foglio {sheet}: {e}")
        else:
            print(f"Colonna 'MAPE' non trovata nel foglio {sheet}.")
    
    if all_mape_values:
        overall_average = sum(all_mape_values) / len(all_mape_values)
        overall_percentage = overall_average * 100  # Conversione in percentuale
        print("\n==============================")
        print(f"Media percentuale complessiva del valore MAPE: {overall_percentage:.2f}%")
        print("==============================\n")
        return overall_percentage
    else:
        print("Nessun valore MAPE trovato nei fogli specificati.")
        return None

if __name__ == "__main__":
    # 1. Calcolo della media del MAPE (in percentuale) dai file di log degli esperimenti
    print("Calcolo della media del MAPE (in percentuale) dai file di log degli esperimenti:")
    compute_average_mape_from_experiment_logs()
    
    # 2. Calcolo della media (non in percentuale) per ogni colonna dei fogli battery_mean_*
    root = tk.Tk()
    root.withdraw()
    excel_file_sheets = filedialog.askopenfilename(
        title="Seleziona il file Excel con i fogli battery_mean_*",
        filetypes=[("Excel files", "*.xlsx;*.xls")]
    )
    
    if excel_file_sheets:
        sheet_averages = compute_sheet_mape_averages(excel_file_sheets)
        if sheet_averages:
            print("Media (non in percentuale) per ogni colonna dei fogli specificati:")
            for sheet, avgs in sheet_averages.items():
                print(f"\nFoglio: {sheet}")
                for col, avg in avgs.items():
                    print(f"  Colonna {col}: {avg:.4f}")
        
        # 3. Calcolo della media percentuale complessiva del valore MAPE dai fogli specificati
        compute_overall_mape_percentage(excel_file_sheets)
    else:
        print("Nessun file Excel selezionato per i fogli battery_mean_*.")
