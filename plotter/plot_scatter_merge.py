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

# Ottiene tutte le sottocartelle (ciascuna rappresenta un esperimento)
subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
if not subdirs:
    print("Nessuna sottocartella trovata in:", folder_path)
    exit(1)

# Liste per raccogliere i valori dei dati reali e predetti
true_all = []
pred_all = []

for subdir in subdirs:
    subdir_path = os.path.join(folder_path, subdir)
    true_file = os.path.join(subdir_path, "true_label.npy")
    pred_file = os.path.join(subdir_path, "pred_label.npy")
    
    if not (os.path.exists(true_file) and os.path.exists(pred_file)):
        print(f"La cartella '{subdir}' non contiene entrambi i file richiesti ('true_label.npy' e 'pred_label.npy').")
        continue

    # Carica i dati
    true_data = np.load(true_file)
    pred_data = np.load(pred_file)
    
    # Aggiunge i dati alle liste complessive
    true_all.extend(true_data)
    pred_all.extend(pred_data)

true_all = np.array(true_all)
pred_all = np.array(pred_all)

# Creazione dello scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(true_all, pred_all, alpha=0.6, label="Esperimenti")
plt.xlabel("True SOH")
plt.ylabel("Prediction")
plt.title("Scatter Plot: Prediction vs True SOH")

# Linea di riferimento ideale y = x
min_val = min(np.min(true_all), np.min(pred_all))
max_val = max(np.max(true_all), np.max(pred_all))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

plt.legend()
plt.grid(True)
plt.show()
