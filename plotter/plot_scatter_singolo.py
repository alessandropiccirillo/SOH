import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Nasconde la finestra principale di Tkinter e apre il dialogo per selezionare la cartella dell'esperimento
root = tk.Tk()
root.withdraw()
exp_folder =filedialog.askdirectory(title=r"C:\Users\Aless\OneDrive\Desktop\PINN4SOH-main\PINN4SOH-main\results of reviewer\XJTU resultes")

if not exp_folder:
    print("Nessuna cartella selezionata. Uscita.")
    exit(1)

# Definisce i percorsi dei file all'interno della cartella selezionata
true_file = os.path.join(exp_folder, "true_label.npy")
pred_file = os.path.join(exp_folder, "pred_label.npy")

if not (os.path.exists(true_file) and os.path.exists(pred_file)):
    print("La cartella selezionata non contiene i file 'true_label.npy' e 'pred_label.npy'.")
    exit(1)

# Carica i dati
true_data = np.load(true_file)
pred_data = np.load(pred_file)

# Creazione dello scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(true_data, pred_data, alpha=0.6, label="Dati")
plt.xlabel("True SOH")
plt.ylabel("Prediction")
plt.title("Scatter Plot: Prediction vs True SOH")

# Aggiunge la linea ideale y = x come riferimento
min_val = min(np.min(true_data), np.min(pred_data))
max_val = max(np.max(true_data), np.max(pred_data))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

plt.legend()
plt.grid(True)
plt.show()
