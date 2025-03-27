import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Nascondi la finestra principale di Tkinter e apri il dialogo per la selezione della cartella
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title=r"C:\Users\Aless\OneDrive\Desktop\PINN4SOH-main\PINN4SOH-main\results of reviewer\XJTU resultes")


if not folder_path:
    print("Nessuna cartella selezionata. Uscita.")
    exit(1)

# Elenca tutte le sottocartelle all'interno della cartella selezionata
subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

if not subdirs:
    print("Nessuna sottocartella trovata in:", folder_path)
    exit(1)

# Itera sulle sottocartelle (cioè sugli esperimenti)
for subdir in subdirs:
    subdir_path = os.path.join(folder_path, subdir)
    true_file = os.path.join(subdir_path, "true_label.npy")
    pred_file = os.path.join(subdir_path, "pred_label.npy")
    
    # Controlla se entrambi i file esistono nella sottocartella
    if not (os.path.exists(true_file) and os.path.exists(pred_file)):
        print(f"La cartella '{subdir}' non contiene entrambi i file richiesti ('true_label.npy' e 'pred_label.npy').")
        continue

    # Carica i dati dai file .npy
    true = np.load(true_file)
    pred = np.load(pred_file)

    # Crea il grafico per l'esperimento corrente
    plt.figure(figsize=(10, 6))
    plt.plot(true, label='True Labels', marker='o')
    plt.plot(pred, label='Predicted Labels', marker='x')
    plt.xlabel('Indice')
    plt.ylabel('Valore')
    plt.title(f'Esperimento: {subdir}')
    plt.legend()
    plt.grid(True)
    plt.show()
