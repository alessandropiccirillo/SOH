import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Nasconde la finestra principale di Tkinter e apre il dialogo per selezionare la cartella dell'esperimento
root = tk.Tk()
root.withdraw()
exp_folder = filedialog.askdirectory(title="Seleziona la cartella dell'esperimento")

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

# Calcola il minimo e massimo globale per la normalizzazione
global_min = min(np.min(true_data), np.min(pred_data))
global_max = max(np.max(true_data), np.max(pred_data))

# Applica la normalizzazione: porta i dati nell'intervallo [-1, 1]
true_data_norm = 2 * (true_data - global_min) / (global_max - global_min) - 1
pred_data_norm = 2 * (pred_data - global_min) / (global_max - global_min) - 1

# Creazione dello scatter plot per i dati normalizzati
plt.figure(figsize=(8, 6))
plt.scatter(true_data_norm, pred_data_norm, alpha=0.6, label="Dati normalizzati")
plt.xlabel("True SOH normalizzato")
plt.ylabel("Prediction normalizzata")
plt.title("Scatter Plot Normalizzato: Prediction vs True SOH")

# Aggiunge la linea ideale y = x come riferimento, che in questo caso va da -1 a 1
plt.plot([-1, 1], [-1, 1], 'r--', label="y = x")

plt.legend()
plt.grid(True)
plt.show()
