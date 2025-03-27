import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Nasconde la finestra principale di Tkinter e apre il dialogo per selezionare la cartella
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title=r"C:\Users\Aless\OneDrive\Desktop\PINN4SOH-main\PINN4SOH-main\results of reviewer\XJTU resultes")

if not folder_path:
    print("Nessuna cartella selezionata. Uscita.")
    exit(1)

# Ottiene tutte le sottocartelle (considerate come esperimenti)
subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
if not subdirs:
    print("Nessuna sottocartella trovata in:", folder_path)
    exit(1)

# Liste per raccogliere gli array degli esperimenti
true_list = []
pred_list = []

for subdir in subdirs:
    subdir_path = os.path.join(folder_path, subdir)
    true_file = os.path.join(subdir_path, "true_label.npy")
    pred_file = os.path.join(subdir_path, "pred_label.npy")
    
    if not (os.path.exists(true_file) and os.path.exists(pred_file)):
        print(f"La cartella '{subdir}' non contiene entrambi i file richiesti ('true_label.npy' e 'pred_label.npy').")
        continue

    # Carica i dati
    true = np.load(true_file)
    pred = np.load(pred_file)
    
    true_list.append(true)
    pred_list.append(pred)

if not true_list or not pred_list:
    print("Nessun esperimento valido trovato.")
    exit(1)

# Supponiamo che tutti gli array abbiano la stessa lunghezza: calcoliamo la media per ogni indice
true_merged = np.mean(true_list, axis=0)
pred_merged = np.mean(pred_list, axis=0)

# Crea l'asse x basato sulla lunghezza degli array
x = np.arange(len(true_merged))

# Plot del merge degli esperimenti in un unico grafico
plt.figure(figsize=(10, 6))
plt.plot(x, true_merged, marker='o', label='True Labels (Media)')
plt.plot(x, pred_merged, marker='x', label='Predicted Labels (Media)')
plt.xlabel('Indice')
plt.ylabel('Valore')
plt.title('Merge degli esperimenti: confronto tra True e Predicted Labels')
plt.legend()
plt.grid(True)
plt.show()
