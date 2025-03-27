import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Nasconde la finestra principale di Tkinter e apre il dialogo per selezionare la cartella degli esperimenti
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title=r"C:\Users\Aless\OneDrive\Desktop\PINN4SOH-main\PINN4SOH-main\results of reviewer\XJTU resultes")


if not folder_path:
    print("Nessuna cartella selezionata. Uscita.")
    exit(1)

# Ottiene tutte le sottocartelle (ognuna rappresenta un esperimento)
subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
if not subdirs:
    print("Nessuna sottocartella trovata in:", folder_path)
    exit(1)

# Liste per raccogliere gli array normalizzati degli esperimenti
true_norm_list = []
pred_norm_list = []

for subdir in subdirs:
    subdir_path = os.path.join(folder_path, subdir)
    true_file = os.path.join(subdir_path, "true_label.npy")
    pred_file = os.path.join(subdir_path, "pred_label.npy")
    
    if not (os.path.exists(true_file) and os.path.exists(pred_file)):
        print(f"La cartella '{subdir}' non contiene entrambi i file ('true_label.npy' e 'pred_label.npy').")
        continue

    # Carica i dati
    true = np.load(true_file)
    pred = np.load(pred_file)
    
    # Calcola min e max per ciascun esperimento e normalizza in [-1, 1]:
    # x_norm = 2*(x - min) / (max - min) - 1
    true_min, true_max = true.min(), true.max()
    pred_min, pred_max = pred.min(), pred.max()
    
    if true_max - true_min != 0:
        true_norm = 2 * (true - true_min) / (true_max - true_min) - 1
    else:
        true_norm = true  # o eventualmente np.zeros_like(true)
    
    if pred_max - pred_min != 0:
        pred_norm = 2 * (pred - pred_min) / (pred_max - pred_min) - 1
    else:
        pred_norm = pred
    
    true_norm_list.append(true_norm)
    pred_norm_list.append(pred_norm)

if not true_norm_list or not pred_norm_list:
    print("Nessun esperimento valido trovato.")
    exit(1)

# Supponiamo che tutti gli array abbiano la stessa lunghezza: calcoliamo la media elemento per elemento
true_merged = np.mean(true_norm_list, axis=0)
pred_merged = np.mean(pred_norm_list, axis=0)

# Asse x basato sugli indici (supponiamo una dimensione comune)
x = np.arange(len(true_merged))

plt.figure(figsize=(10, 6))
plt.plot(x, true_merged, marker='o', label='True Labels (Min-Max normalizzati in [-1,1])')
plt.plot(x, pred_merged, marker='x', label='Predicted Labels (Min-Max normalizzati in [-1,1])')
plt.xlabel('Indice (campione)')
plt.ylabel('Valore normalizzato')
plt.title('Merge degli esperimenti con Normalizzazione Min-Max in [-1, 1]')
plt.legend()
plt.grid(True)
plt.show()
