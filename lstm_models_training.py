# Bloc 1: Imports et Paramètres Globaux pour l'IA
# Paramètres pour la préparation des séquences et la prédiction

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Bloc 1: Paramètres Globaux ---

WINDOW_SIZE = 150  # 3 secondes * 50 Hz
PREDICTION_HORIZON = 25
FEATURES_TO_USE = [
    'Acc_X_Raw (m/s^2)', 'Acc_Y_Raw (m/s^2)', 'Acc_Z_Raw (m/s^2)',
    'Gyro_X (rad/s)', 'Gyro_Y (rad/s)', 'Gyro_Z (rad/s)',
    'Force_Cable1 (N)', 'Force_Cable2 (N)',
    'Vel_X_Noisy (m/s)', 'Vel_Y_Noisy (m/s)', 'Vel_Z_Noisy (m/s)',
    'Angle_Pitch_Noisy (rad)', 'Angle_Yaw_Noisy (rad)'
]
SAMPLING_RATE = 50
DT = 1.0 / SAMPLING_RATE

BASE_MODEL_FILENAME = "deblocage_model"
BASE_SCALER_FILENAME = "scaler"
TRAINING_DATA_DIR = "training_dataset"
INFERENCE_DATA_DIR = "inference_dataset"

PREDICTION_THRESHOLD_ANALYSIS = 0.7
PERSISTENCE_STEPS_ANALYSIS = int(0.1 * SAMPLING_RATE)
TOLERANCE_EARLY_S_ANALYSIS = 0.3
TOLERANCE_LATE_S_ANALYSIS = 0.7
MAX_ACCEPTABLE_DELAY_S_ANALYSIS = 2.0

# --- Bloc 2: Fonctions de Chargement et de Prétraitement des Données ---
DEBUG_FILE_COUNT_LOAD_PROCESS = 0
MAX_DEBUG_FILES_LOAD_PROCESS = 2 # Mettre à 0 pour désactiver le debug détaillé du chargement

def get_second_click_time(button_states, sampling_rate_local):
    clicks_start_indices = np.where(np.diff(button_states.astype(int)) == 1)[0] + 1
    if len(button_states) > 0 and button_states[0] == 1:
        if not (len(clicks_start_indices) > 0 and clicks_start_indices[0] == 0):
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)
        elif not np.any(clicks_start_indices == 0):
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)
    if len(clicks_start_indices) >= 2:
        return clicks_start_indices[1], clicks_start_indices[1] / sampling_rate_local
    return None, None

def create_sequences_and_labels_for_file(df, window_size, prediction_horizon, features, mode="training", sampling_rate_local=50, filename_for_debug="Unknown"):
    global DEBUG_FILE_COUNT_LOAD_PROCESS
    X, y = [], []
    raw_button_state = df['Button_State'].values.astype(int)
    actual_second_click_idx, actual_second_click_time = None, None

    if DEBUG_FILE_COUNT_LOAD_PROCESS < MAX_DEBUG_FILES_LOAD_PROCESS and mode == "training":
        print(f"\n[Debug create_sequences] Fichier: {filename_for_debug}, Mode: {mode}")
        s_idx_dbg, s_time_dbg = get_second_click_time(raw_button_state, sampling_rate_local)
        if s_idx_dbg is not None: print(f"[Debug create_sequences] 2e clic trouvé à l'index {s_idx_dbg} (temps {s_time_dbg:.2f}s)")
        else: print(f"[Debug create_sequences] PAS de 2e clic trouvé.")

    if mode == "training":
        actual_second_click_idx, actual_second_click_time = get_second_click_time(raw_button_state, sampling_rate_local)
        if actual_second_click_idx is None and DEBUG_FILE_COUNT_LOAD_PROCESS < MAX_DEBUG_FILES_LOAD_PROCESS:
            print(f"Avertissement [Fichier: {filename_for_debug}]: Aucun 2e clic pour étiquetage. Étiquettes seront 0.")

    data_for_sequences = df[features].copy()
    positive_labels_found_in_file = 0
    for i in range(len(data_for_sequences) - window_size - prediction_horizon + 1):
        input_seq = data_for_sequences.iloc[i : i + window_size].values
        label = 0
        if mode == "training" and actual_second_click_idx is not None:
            start_predict_window_idx = i + window_size
            end_predict_window_idx = i + window_size + prediction_horizon
            if start_predict_window_idx <= actual_second_click_idx < end_predict_window_idx:
                label = 1
                positive_labels_found_in_file +=1
        X.append(input_seq)
        y.append(label)

    if DEBUG_FILE_COUNT_LOAD_PROCESS < MAX_DEBUG_FILES_LOAD_PROCESS and mode == "training":
        print(f"[Debug create_sequences] Fichier {filename_for_debug}: {positive_labels_found_in_file} étiquettes positives (1) créées.")
        DEBUG_FILE_COUNT_LOAD_PROCESS += 1
    return np.array(X), np.array(y), actual_second_click_time

def load_process_all_simulations(main_directory, mode, window_size, prediction_horizon, features, sampling_rate_local=50, scaler_obj=None):
    all_X, all_y, all_original_indices_and_files = [], [], []
    print(f"\nAnalyse du répertoire : {os.path.abspath(main_directory)} pour le mode '{mode}'")
    if not os.path.isdir(main_directory):
        print(f"Erreur : Le répertoire '{main_directory}' n'existe pas."); return np.array([]), np.array([]), [], None

    file_pattern_suffix = f"_{mode}.xlsx"
    simulation_files = [f for f in os.listdir(main_directory) if f.endswith(file_pattern_suffix) and os.path.isfile(os.path.join(main_directory, f))]

    if mode == "training" and not simulation_files:
        print(f"Aucun fichier '{file_pattern_suffix}' trouvé. Recherche de fichiers '.xlsx' génériques pour le training...")
        simulation_files = [f for f in os.listdir(main_directory)
                            if f.endswith('.xlsx') and 'inference' not in f.lower() and 'full_for_plot' not in f.lower()
                            and os.path.isfile(os.path.join(main_directory, f))]

    print(f"Fichiers trouvés correspondant au mode '{mode}': {simulation_files if simulation_files else 'Aucun'}")
    if not simulation_files: return np.array([]), np.array([]), [], None

    file_count = 0
    for filename in simulation_files:
        filepath = os.path.join(main_directory, filename)
        try:
            df = pd.read_excel(filepath)
            file_count += 1
            if not all(feat in df.columns for feat in features):
                print(f"Avertissement: Caractéristiques manquantes dans {filename}. Ignoré."); continue
            if 'Button_State' not in df.columns:
                print(f"Avertissement: 'Button_State' manquant dans {filename}. Ignoré."); continue

            X_file, y_file, second_click_time_file = create_sequences_and_labels_for_file(
                df, window_size, prediction_horizon, features, mode, sampling_rate_local, filename_for_debug=filename)
            if X_file.shape[0] > 0:
                all_X.append(X_file)
                if mode == "training": all_y.append(y_file)
                for seq_idx in range(X_file.shape[0]):
                    all_original_indices_and_files.append({
                        "filename": filename, "sequence_index_in_file": seq_idx,
                        "prediction_window_start_time": (seq_idx + window_size) / sampling_rate_local,
                        "actual_second_click_time": second_click_time_file if mode == "training" else None})
        except Exception as e: print(f"Erreur traitant {filename}: {e}")

    print(f"{file_count} fichiers traités dans {main_directory}.")
    if not all_X: print("Aucune séquence créée."); return np.array([]), np.array([]), [], None

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0) if mode == "training" and all_y else np.array([])
    if X_combined.shape[0] == 0: return np.array([]), np.array([]), [], None

    num_samples, num_timesteps, num_features = X_combined.shape
    X_reshaped = X_combined.reshape(-1, num_features)
    current_scaler = scaler_obj
    if mode == "training":
        print("Ajustement du scaler..."); current_scaler = StandardScaler(); X_scaled_reshaped = current_scaler.fit_transform(X_reshaped); print("Scaler ajusté.")
    elif current_scaler is not None:
        print("Transformation avec scaler fourni..."); X_scaled_reshaped = current_scaler.transform(X_reshaped); print("Données transformées.")
    else:
        print("Avertissement: Pas de scaler pour l'inférence."); X_scaled_reshaped = X_reshaped
    X_scaled = X_scaled_reshaped.reshape(num_samples, num_timesteps, num_features)
    return X_scaled, y_combined, all_original_indices_and_files, current_scaler

# --- Bloc 3: Définitions des Modèles ---
def build_lstm_model(input_shape):
    model = Sequential(name="LSTM_Model")
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("\n--- Architecture du Modèle LSTM ---")
    model.summary()
    return model

def build_cnn_lstm_model(input_shape):
    model = Sequential(name="CNN_LSTM_Model")
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("\n--- Architecture du Modèle CNN-LSTM ---")
    model.summary()
    return model

# --- Bloc 4: Entraînement du Modèle ---
print("\n--- Phase d'Entraînement ---")
model_type_to_train = ""
while model_type_to_train not in ["lstm", "cnn-lstm"]:
    model_type_to_train = input("Quel type de modèle entraîner ('lstm' ou 'cnn-lstm') ? ").lower()
print(f"Choix pour l'entraînement : Modèle {model_type_to_train.upper()}")

model_filepath_train = f"{BASE_MODEL_FILENAME}_{model_type_to_train}.h5"
scaler_filepath_train = f"{BASE_SCALER_FILENAME}_{model_type_to_train}.joblib"

DEBUG_FILE_COUNT_LOAD_PROCESS = 0
X_train_full, y_train_full, train_mapping_info, scaler_train = load_process_all_simulations(
    TRAINING_DATA_DIR, mode="training", window_size=WINDOW_SIZE,
    prediction_horizon=PREDICTION_HORIZON, features=FEATURES_TO_USE, sampling_rate_local=SAMPLING_RATE
)

if X_train_full.shape[0] > 0 and y_train_full.shape[0] > 0:
    joblib.dump(scaler_train, scaler_filepath_train)
    print(f"Scaler sauvegardé dans {scaler_filepath_train}")

    can_stratify_train = len(np.unique(y_train_full)) > 1
    X_train, X_val, y_train, y_val, mapping_train, mapping_val = train_test_split(
        X_train_full, y_train_full, train_mapping_info, # Inclure le mapping dans le split
        test_size=0.2, shuffle=True, stratify=y_train_full if can_stratify_train else None
    )

    print(f"Entraînement: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")
    if len(y_train) > 0: print(f"Distr. y_train: {np.bincount(y_train.astype(int))}")
    if len(y_val) > 0: print(f"Distr. y_val: {np.bincount(y_val.astype(int))}")

    has_pos_train = len(y_train) > 0 and np.sum(y_train) > 0

    if X_train.shape[0] > 0 and X_val.shape[0] > 0 and has_pos_train:
        print(f"\nConstruction et entraînement du modèle : {model_type_to_train.upper()}")
        if model_type_to_train == "lstm":
            model_to_train = build_lstm_model(input_shape=(WINDOW_SIZE, X_train.shape[2]))
        else:
            model_to_train = build_cnn_lstm_model(input_shape=(WINDOW_SIZE, X_train.shape[2]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_filepath_train, monitor='val_accuracy', save_best_only=True, verbose=1)

        count_0, count_1 = np.sum(y_train == 0), np.sum(y_train == 1)
        class_w_train = {0: (1/count_0)*(len(y_train)/2.0) if count_0>0 else 1, 1: (1/count_1)*(len(y_train)/2.0) if count_1>0 else 1}
        print(f"Poids des classes pour l'entraînement: {class_w_train}")

        history = model_to_train.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val),
                                 callbacks=[early_stopping, model_checkpoint], class_weight=class_w_train)
        print(f"Modèle {model_type_to_train} entraîné et meilleur modèle sauvegardé dans {model_filepath_train}")

        print(f"\n--- Performance finale du modèle {model_type_to_train.upper()} sur l'ensemble de validation (après entraînement) ---")
        val_loss, val_accuracy = model_to_train.evaluate(X_val, y_val, verbose=0)
        print(f"Perte de validation finale: {val_loss:.4f}")
        print(f"Précision de validation finale: {val_accuracy:.4f}")

        plt.figure(figsize=(12, 4)); plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Perte (train)'); plt.plot(history.history['val_loss'], label='Perte (val)'); plt.title(f'Perte - {model_type_to_train.upper()}'); plt.xlabel('Époque'); plt.ylabel('Perte'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2); plt.plot(history.history['accuracy'], label='Précision (train)'); plt.plot(history.history['val_accuracy'], label='Précision (val)'); plt.title(f'Précision - {model_type_to_train.upper()}'); plt.xlabel('Époque'); plt.ylabel('Précision'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.show()
    else: print("Données d'entraînement/validation insuffisantes ou pas d'exemples positifs.")
else: print(f"Aucune donnée chargée depuis {TRAINING_DATA_DIR} pour le mode training.")

# --- Bloc 5: Analyse des Erreurs de Prédiction ---
def find_predicted_deblocage_event_time_analysis(probabilities, window_start_times, threshold, persistence_steps):
    if probabilities is None or len(probabilities) < persistence_steps: return None
    for i in range(len(probabilities) - persistence_steps + 1):
        if np.all(probabilities[i : i + persistence_steps] > threshold):
            return window_start_times[i]
    return None

print("\n--- Phase d'Analyse des Erreurs de Prédiction ---")
model_type_to_analyze = ""
while model_type_to_analyze not in ["lstm", "cnn-lstm"]:
    model_type_to_analyze = input("Quel type de modèle analyser ('lstm' ou 'cnn-lstm') ? ").lower()
print(f"Choix pour l'analyse : Modèle {model_type_to_analyze.upper()}")

model_filepath_analysis = f"{BASE_MODEL_FILENAME}_{model_type_to_analyze}.h5"
scaler_filepath_analysis = f"{BASE_SCALER_FILENAME}_{model_type_to_analyze}.joblib"

if not (os.path.exists(model_filepath_analysis) and os.path.exists(scaler_filepath_analysis)):
    print(f"Modèle ({model_filepath_analysis}) ou Scaler ({scaler_filepath_analysis}) non trouvé pour l'analyse.")
else:
    model_analysis = load_model(model_filepath_analysis)
    scaler_analysis = joblib.load(scaler_filepath_analysis)
    print(f"Modèle et scaler chargés pour l'analyse ({model_type_to_analyze}).")

    DEBUG_FILE_COUNT_LOAD_PROCESS = 0
    # Utiliser les données de validation (X_val, y_val, mapping_val) si elles existent et ont été générées
    # Sinon, recharger toutes les données d'entraînement pour l'analyse
    X_data_for_analysis, y_data_for_analysis, mapping_for_analysis = None, None, None
    if 'X_val' in locals() and X_val.shape[0] > 0 and 'y_val' in locals() and 'mapping_val' in locals():
        print("Utilisation de l'ensemble de validation précédent pour l'analyse des erreurs.")
        X_data_for_analysis = X_val
        y_data_for_analysis = y_val # y_val contient les étiquettes de fenêtre
        mapping_for_analysis = mapping_val # mapping_val contient les infos pour retrouver les temps de clic réels
    else:
        print("Ensemble de validation non disponible. Rechargement des données d'entraînement pour l'analyse des erreurs...")
        X_data_for_analysis, y_data_for_analysis, mapping_for_analysis, _ = load_process_all_simulations(
            TRAINING_DATA_DIR, mode="training", window_size=WINDOW_SIZE,
            prediction_horizon=PREDICTION_HORIZON, features=FEATURES_TO_USE,
            sampling_rate_local=SAMPLING_RATE, scaler_obj=scaler_analysis)

    if X_data_for_analysis is not None and X_data_for_analysis.shape[0] > 0:
        print(f"Données pour analyse chargées: {X_data_for_analysis.shape}")
        probabilities_analysis = model_analysis.predict(X_data_for_analysis).flatten()
        y_pred_analysis_window = (probabilities_analysis > 0.5).astype(int) # Prédictions par fenêtre

        print(f"\n--- Rapport de Classification (par fenêtre) pour Modèle {model_type_to_analyze.upper()} sur les données d'analyse ---")
        if y_data_for_analysis is not None and len(y_data_for_analysis) == len(y_pred_analysis_window):
            print(classification_report(y_data_for_analysis, y_pred_analysis_window, zero_division=0))
            cm_analysis = confusion_matrix(y_data_for_analysis, y_pred_analysis_window)
            plt.figure(figsize=(6,4)); sns.heatmap(cm_analysis, annot=True, fmt='d', cmap='Blues', xticklabels=['Pas Clic', 'Clic'], yticklabels=['Pas Clic', 'Clic'])
            plt.title(f'Matrice de Confusion (Fenêtres) - {model_type_to_analyze.upper()}'); plt.ylabel('Vrai'); plt.xlabel('Prédit'); plt.show()
        else:
            print("Étiquettes réelles (y_data_for_analysis) non disponibles ou de taille incorrecte pour le rapport de classification par fenêtre.")

        time_differences_analysis = []
        categories_analysis = {"hit": 0, "early": 0, "late": 0, "missed_event": 0, "no_true_event_in_mapping": 0}

        file_to_indices_analysis = {}
        for i, mapping in enumerate(mapping_for_analysis):
            file_to_indices_analysis.setdefault(mapping["filename"], []).append(i)

        for filename, indices in file_to_indices_analysis.items():
            if not indices: continue # Au cas où une liste d'indices serait vide
            actual_click_time_for_file = mapping_for_analysis[indices[0]]["actual_second_click_time"]

            if actual_click_time_for_file is None:
                categories_analysis["no_true_event_in_mapping"] += 1; continue

            file_probabilities = probabilities_analysis[indices]
            file_prediction_window_start_times = [mapping_for_analysis[i]["prediction_window_start_time"] for i in indices]

            predicted_time = find_predicted_deblocage_event_time_analysis(
                file_probabilities, file_prediction_window_start_times,
                PREDICTION_THRESHOLD_ANALYSIS, PERSISTENCE_STEPS_ANALYSIS
            )

            if predicted_time is not None:
                diff = predicted_time - actual_click_time_for_file
                time_differences_analysis.append(diff)
                if diff < (-TOLERANCE_EARLY_S_ANALYSIS): categories_analysis["early"] += 1
                elif diff > TOLERANCE_LATE_S_ANALYSIS:
                    if diff <= MAX_ACCEPTABLE_DELAY_S_ANALYSIS: categories_analysis["late"] += 1
                    else: categories_analysis["missed_event"] +=1
                else: categories_analysis["hit"] += 1
            else: categories_analysis["missed_event"] += 1

        if time_differences_analysis:
            time_differences_analysis = np.array(time_differences_analysis)
            plt.figure(figsize=(10, 6)); sns.histplot(time_differences_analysis, kde=True, bins=30)
            plt.title(f"Distribution Écarts Temporels ({model_type_to_analyze})"); plt.xlabel("Écart (s)"); plt.ylabel("Nombre"); plt.grid(True, linestyle='--')
            plt.axvline(-TOLERANCE_EARLY_S_ANALYSIS, color='orange', ls='--', label=f'Trop Tôt (-{TOLERANCE_EARLY_S_ANALYSIS}s)'); plt.axvline(TOLERANCE_LATE_S_ANALYSIS, color='red', ls='--', label=f'Trop Tard (+{TOLERANCE_LATE_S_ANALYSIS}s)')
            plt.axvline(0, color='green', ls='-', label='Parfait'); plt.legend(); plt.show()

            print(f"\nStats Écarts ({model_type_to_analyze}): Moy={np.mean(time_differences_analysis):.3f}s, Med={np.median(time_differences_analysis):.3f}s, Std={np.std(time_differences_analysis):.3f}s")
            total_eval = sum(categories_analysis[k] for k in ["hit", "early", "late", "missed_event"])
            if total_eval > 0:
                for cat, count in categories_analysis.items():
                    if cat != "no_true_event_in_mapping": print(f"  {cat.capitalize()}: {count} ({count/total_eval*100:.1f}%)")
            if categories_analysis["no_true_event_in_mapping"] > 0: print(f"  Fichiers sans 2e clic réel dans mapping: {categories_analysis['no_true_event_in_mapping']}")
        else: print("Aucun écart temporel calculé pour l'analyse des erreurs de timing.")
    else: print(f"Aucune donnée de test chargée pour l'analyse des erreurs.")

# --- Bloc 6: Inférence ---
print("\n--- Phase d'Inférence ---")
model_type_to_infer = ""
while model_type_to_infer not in ["lstm", "cnn-lstm"]:
    model_type_to_infer = input("Quel type de modèle utiliser pour l'inférence ('lstm' ou 'cnn-lstm') ? ").lower()
print(f"Choix pour l'inférence : Modèle {model_type_to_infer.upper()}")

model_filepath_infer = f"{BASE_MODEL_FILENAME}_{model_type_to_infer}.h5"
scaler_filepath_infer = f"{BASE_SCALER_FILENAME}_{model_type_to_infer}.joblib"

if os.path.exists(model_filepath_infer) and os.path.exists(scaler_filepath_infer):
    inference_model = load_model(model_filepath_infer)
    scaler_for_inference = joblib.load(scaler_filepath_infer)
    print(f"Modèle {model_type_to_infer} et scaler chargés pour l'inférence.")

    DEBUG_FILE_COUNT_LOAD_PROCESS = 0
    X_inference, _, inference_mapping_info, _ = load_process_all_simulations(
        INFERENCE_DATA_DIR, mode="inference", window_size=WINDOW_SIZE,
        prediction_horizon=PREDICTION_HORIZON, features=FEATURES_TO_USE,
        sampling_rate_local=SAMPLING_RATE, scaler_obj=scaler_for_inference)

    if X_inference.shape[0] > 0:
        print(f"Données d'inférence chargées: {X_inference.shape}")
        predictions_proba_inference = inference_model.predict(X_inference).flatten()

        first_inf_file = inference_mapping_info[0]["filename"] if inference_mapping_info else None
        indices_first_inf_file = [i for i, m in enumerate(inference_mapping_info) if m["filename"] == first_inf_file]

        if indices_first_inf_file and first_inf_file:
            start_idx, end_idx = indices_first_inf_file[0], indices_first_inf_file[-1] + 1
            pred_times_plot = [m["prediction_window_start_time"] for m in inference_mapping_info[start_idx:end_idx]]

            plt.figure(figsize=(15, 6)); plt.plot(pred_times_plot, predictions_proba_inference[start_idx:end_idx], label=f"Proba. Déblocage ({model_type_to_infer} - {first_inf_file})")
            plt.axhline(0.5, color='r', ls='--', label="Seuil 0.5"); plt.title(f"Prédictions Inférence: {first_inf_file}"); plt.xlabel("Temps (s)"); plt.ylabel("Probabilité"); plt.legend(); plt.grid(True); plt.ylim(0,1); plt.show()

            predicted_inf_time = find_predicted_deblocage_event_time_analysis(
                predictions_proba_inference[start_idx:end_idx], pred_times_plot,
                PREDICTION_THRESHOLD_ANALYSIS, PERSISTENCE_STEPS_ANALYSIS)
            if predicted_inf_time: print(f"Événement déblocage PRÉDIT (inférence {first_inf_file}) à ~{predicted_inf_time:.2f}s")
            else: print(f"Aucun événement déblocage prédit (inférence {first_inf_file})")
        else: print("Impossible d'isoler prédictions pour 1er fichier d'inférence ou pas d'info de mapping.")
    else: print(f"Aucune donnée chargée depuis {INFERENCE_DATA_DIR} pour le mode inference.")
else: print(f"Modèle ou scaler non trouvé pour {model_type_to_infer}. Entraînez d'abord.")

