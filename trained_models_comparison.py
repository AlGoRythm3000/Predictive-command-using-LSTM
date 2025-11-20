# --- Paramètres Globaux (Rappel, doivent être cohérents) ---
import time # Ajouter l'import de time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score # Ajouter l'import de precision_recall_fscore_support et accuracy_score
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

WINDOW_SIZE_COMP = 150
FEATURES_TO_USE_COMP = [
    'Acc_X_Raw (m/s^2)', 'Acc_Y_Raw (m/s^2)', 'Acc_Z_Raw (m/s^2)',
    'Gyro_X (rad/s)', 'Gyro_Y (rad/s)', 'Gyro_Z (rad/s)',
    'Force_Cable1 (N)', 'Force_Cable2 (N)',
    'Vel_X_Noisy (m/s)', 'Vel_Y_Noisy (m/s)', 'Vel_Z_Noisy (m/s)',
    'Angle_Pitch_Noisy (rad)', 'Angle_Yaw_Noisy (rad)'
]
SAMPLING_RATE_COMP = 50
PREDICTION_HORIZON_COMP = 25 # Important pour la fonction create_sequences...

# Chemins de base (les noms de fichiers seront complétés par _lstm ou _cnn-lstm)
BASE_MODEL_FILENAME_COMP = "deblocage_model"
BASE_SCALER_FILENAME_COMP = "scaler"
# Utiliser le même répertoire de données d'entraînement pour charger un ensemble de test/validation cohérent
DATA_DIR_FOR_COMP = "training_dataset"

# Nombre de répétitions pour mesurer le temps d'inférence
N_INFERENCE_REPEATS = 10
INFERENCE_BATCH_SIZE = 256 # Taille de batch pour l'inférence

# --- Fonctions Utilitaires (simplifiées pour le chargement de données de test) ---
def get_second_click_time_comp(button_states, sampling_rate_local):
    clicks_start_indices = np.where(np.diff(button_states.astype(int)) == 1)[0] + 1
    if len(button_states) > 0 and button_states[0] == 1:
        if not (len(clicks_start_indices) > 0 and clicks_start_indices[0] == 0):
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)
        elif not np.any(clicks_start_indices == 0):
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)
    if len(clicks_start_indices) >= 2:
        return clicks_start_indices[1], clicks_start_indices[1] / sampling_rate_local
    return None, None

def create_sequences_and_labels_for_file_comp(df, window_size, prediction_horizon, features, sampling_rate_local=50):
    X, y = [], []
    raw_button_state = df['Button_State'].values.astype(int)
    actual_second_click_idx, _ = get_second_click_time_comp(raw_button_state, sampling_rate_local)

    data_for_sequences = df[features].copy()
    for i in range(len(data_for_sequences) - window_size - prediction_horizon + 1):
        input_seq = data_for_sequences.iloc[i : i + window_size].values
        label = 0
        if actual_second_click_idx is not None:
            start_predict_window_idx = i + window_size
            end_predict_window_idx = i + window_size + prediction_horizon
            if start_predict_window_idx <= actual_second_click_idx < end_predict_window_idx:
                label = 1
        X.append(input_seq)
        y.append(label)
    return np.array(X), np.array(y)

def load_and_prepare_data_for_comparison(main_directory, window_size, prediction_horizon, features, sampling_rate_local, scaler_to_use):
    all_X_comp, all_y_comp = [], []
    print(f"Chargement des données pour comparaison depuis : {os.path.abspath(main_directory)}")

    simulation_files = [f for f in os.listdir(main_directory)
                        if (f.endswith('_training.xlsx') or (f.endswith('.xlsx') and 'inference' not in f.lower() and 'full_for_plot' not in f.lower()))
                        and os.path.isfile(os.path.join(main_directory, f))]

    if not simulation_files:
        print("Aucun fichier de données trouvé pour la comparaison.")
        return np.array([]), np.array([])

    # Pour être plus robuste, on pourrait prendre un nombre fixe de fichiers ou de séquences
    # ou s'assurer que X_val/y_val de l'entraînement sont passés à cette fonction.
    # Ici, on recharge tout et on resplitte, ce qui est ok pour une comparaison mais peut être long.
    # On va limiter aux 50 premiers fichiers pour la comparaison pour que ce soit plus rapide.
    print(f"Utilisation des {min(len(simulation_files), 50)} premiers fichiers pour la comparaison.")
    files_to_load_for_comp = simulation_files[:min(len(simulation_files), 50)]


    for filename in files_to_load_for_comp:
        filepath = os.path.join(main_directory, filename)
        try:
            df = pd.read_excel(filepath)
            if not all(feat in df.columns for feat in features) or 'Button_State' not in df.columns:
                print(f"Avertissement: Colonnes manquantes dans {filename} pour la comparaison. Ignoré.")
                continue
            X_file, y_file = create_sequences_and_labels_for_file_comp(df, window_size, prediction_horizon, features, sampling_rate_local)
            if X_file.shape[0] > 0:
                all_X_comp.append(X_file)
                all_y_comp.append(y_file)
        except Exception as e: print(f"Erreur chargement {filename} pour comparaison: {e}")

    if not all_X_comp:
        print("Aucune séquence chargée pour la comparaison.")
        return np.array([]), np.array([])

    X_combined = np.concatenate(all_X_comp, axis=0)
    y_combined = np.concatenate(all_y_comp, axis=0)

    if X_combined.shape[0] == 0:
        print("X_combined est vide après concaténation pour la comparaison.")
        return np.array([]), np.array([])

    num_samples, num_timesteps, num_features = X_combined.shape
    X_reshaped = X_combined.reshape(-1, num_features)

    if scaler_to_use is None:
        print("Erreur: Scaler non fourni pour la préparation des données de comparaison.")
        return np.array([]), np.array([])

    X_scaled_reshaped = scaler_to_use.transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(num_samples, num_timesteps, num_features)

    # Utiliser un split fixe pour la reproductibilité de la comparaison
    # Si y_combined a des classes, stratifier.
    if len(np.unique(y_combined)) > 1:
        _, X_test_comp, _, y_test_comp = train_test_split(X_scaled, y_combined, test_size=0.3, shuffle=True, stratify=y_combined, random_state=42)
    else:
        _, X_test_comp, _, y_test_comp = train_test_split(X_scaled, y_combined, test_size=0.3, shuffle=True, random_state=42)

    print(f"Ensemble de données de comparaison préparé: X_shape={X_test_comp.shape}, y_shape={y_test_comp.shape}")
    if X_test_comp.shape[0] == 0:
        print("L'ensemble de test pour la comparaison est vide.")
    return X_test_comp, y_test_comp

# --- Script Principal de Comparaison ---
print("\n--- COMPARAISON DES MODÈLES ENTRAÎNÉS ---")

model_types_to_compare = ["lstm", "cnn-lstm"]
comparison_results = {}

# Charger un ensemble de données commun pour l'évaluation
# On a besoin d'un scaler pour cela. On va prendre le scaler du premier modèle qui existe.
# Idéalement, les données devraient être préparées une seule fois avec un scaler "universel"
# ou chaque modèle est testé sur des données normalisées avec SON PROPRE scaler.
# Pour cette comparaison, nous allons normaliser les données avec le scaler de chaque modèle respectif.

for model_type in model_types_to_compare:
    print(f"\nÉvaluation du modèle : {model_type.upper()}")
    model_filepath_comp = f"{BASE_MODEL_FILENAME_COMP}_{model_type}.h5"
    scaler_filepath_comp = f"{BASE_SCALER_FILENAME_COMP}_{model_type}.joblib"

    if not (os.path.exists(model_filepath_comp) and os.path.exists(scaler_filepath_comp)):
        print(f"Modèle ou scaler pour {model_type} non trouvé à '{model_filepath_comp}' ou '{scaler_filepath_comp}'. Passe à suivant.")
        comparison_results[model_type] = {"error": "Fichiers modèle/scaler non trouvés"}
        continue

    try:
        model_comp = load_model(model_filepath_comp)
        scaler_comp = joblib.load(scaler_filepath_comp)
        print(f"Modèle et scaler {model_type.upper()} chargés.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ou du scaler pour {model_type}: {e}")
        comparison_results[model_type] = {"error": f"Erreur chargement: {e}"}
        continue

    X_eval_comp, y_eval_comp = load_and_prepare_data_for_comparison(
        DATA_DIR_FOR_COMP, WINDOW_SIZE_COMP, PREDICTION_HORIZON_COMP,
        FEATURES_TO_USE_COMP, SAMPLING_RATE_COMP, scaler_comp # Utiliser le scaler du modèle actuel
    )

    if X_eval_comp.shape[0] == 0 or y_eval_comp.shape[0] == 0:
        print(f"Aucune donnée d'évaluation pour {model_type} après chargement/préparation. Passe à suivant.")
        comparison_results[model_type] = {"error": "Aucune donnée d'évaluation préparée"}
        continue

    # 1. Métriques de Classification
    y_pred_proba_comp = model_comp.predict(X_eval_comp, batch_size=INFERENCE_BATCH_SIZE, verbose=0)
    y_pred_comp = (y_pred_proba_comp > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_eval_comp, y_pred_comp)
    # average='binary' suppose que la classe positive est 1.
    # Si y_eval_comp ne contient que des 0, precision/recall/f1 pour la classe 1 seront 0.
    if 1 not in y_eval_comp and 1 not in y_pred_comp: # Si aucun positif ni prédit ni réel
        precision, recall, f1 = 0.0, 0.0, 0.0
    elif 1 not in y_eval_comp: # Si aucun positif réel, mais potentiellement prédit
        precision, recall, f1 = 0.0, 0.0, 0.0 # Recall est 0, Precision peut être 0 si FP > 0
    else: # Il y a des positifs réels
        precision, recall, f1, _ = precision_recall_fscore_support(y_eval_comp, y_pred_comp, average='binary', pos_label=1, zero_division=0)

    print(f"\n--- Métriques de Classification pour {model_type.upper()} (sur fenêtres de l'ensemble de comparaison) ---")
    print(f"  Distribution y_eval_comp: {np.bincount(y_eval_comp.astype(int))}")
    print(f"  Précision Globale (Accuracy): {accuracy:.4f}")
    print(f"  Précision (Classe 1 - Clic): {precision:.4f}")
    print(f"  Rappel (Classe 1 - Clic): {recall:.4f}")
    print(f"  Score F1 (Classe 1 - Clic): {f1:.4f}")

    # 2. Temps d'Inférence Moyen
    inference_times = []
    if X_eval_comp.shape[0] >= INFERENCE_BATCH_SIZE : # S'assurer qu'on a au moins un batch pour l'échauffement
        _ = model_comp.predict(X_eval_comp[:INFERENCE_BATCH_SIZE], batch_size=INFERENCE_BATCH_SIZE, verbose=0)

    num_samples_for_timing = X_eval_comp.shape[0]
    if num_samples_for_timing == 0:
        avg_inference_time_per_sample = float('nan')
        print(f"\n--- Temps d'Inférence pour {model_type.upper()} ---")
        print(f"  Pas assez de données pour mesurer le temps d'inférence.")
    else:
        for _ in range(N_INFERENCE_REPEATS):
            start_time = time.time()
            _ = model_comp.predict(X_eval_comp, batch_size=INFERENCE_BATCH_SIZE, verbose=0)
            end_time = time.time()
            inference_times.append(end_time - start_time)

        avg_inference_time_total_run = np.mean(inference_times)
        avg_inference_time_per_sample = (avg_inference_time_total_run / num_samples_for_timing) * 1000 # en ms

        print(f"\n--- Temps d'Inférence pour {model_type.upper()} ---")
        print(f"  Temps moyen pour traiter {num_samples_for_timing} échantillons: {avg_inference_time_total_run:.4f} s (sur {N_INFERENCE_REPEATS} répétitions)")
        print(f"  Temps d'inférence moyen par échantillon (fenêtre): {avg_inference_time_per_sample:.4f} ms")

    comparison_results[model_type] = {
        "accuracy": accuracy,
        "precision_class1": precision,
        "recall_class1": recall,
        "f1_score_class1": f1,
        "avg_inference_time_ms_per_sample": avg_inference_time_per_sample
    }

# --- Affichage des Résultats de Comparaison ---
print("\n\n--- RÉSUMÉ DE LA COMPARAISON DES MODÈLES ---")
results_df = pd.DataFrame.from_dict(comparison_results, orient='index')
print(results_df.to_string()) # .to_string() pour un meilleur affichage dans les logs

if not results_df.empty:
    metrics_to_plot = ['accuracy', 'precision_class1', 'recall_class1', 'f1_score_class1']
    plot_df = results_df[metrics_to_plot].copy()
    for col in metrics_to_plot: plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

    if not plot_df.dropna().empty:
        plot_df.plot(kind='bar', figsize=(12, 7), rot=0)
        plt.title("Comparaison des Métriques de Classification des Modèles"); plt.ylabel("Score")
        plt.ylim(0, max(1.05, plot_df[metrics_to_plot].max().max() * 1.1 if not plot_df.empty else 1.05) ); plt.grid(axis='y', linestyle='--'); plt.show()
    else:
        print("Aucune donnée numérique valide à plotter pour les métriques de classification.")

    if 'avg_inference_time_ms_per_sample' in results_df.columns:
        time_plot_df = results_df[['avg_inference_time_ms_per_sample']].copy()
        time_plot_df['avg_inference_time_ms_per_sample'] = pd.to_numeric(time_plot_df['avg_inference_time_ms_per_sample'], errors='coerce')

        if not time_plot_df.dropna().empty:
            time_plot_df.plot(kind='bar', figsize=(8, 5), rot=0, legend=False)
            plt.title("Temps d'Inférence Moyen par Échantillon"); plt.ylabel("Temps (ms)")
            plt.grid(axis='y', linestyle='--'); plt.show()
        else:
            print("Pas de données valides de temps d'inférence à plotter.")
else:
    print("Aucun résultat de comparaison à afficher.")
