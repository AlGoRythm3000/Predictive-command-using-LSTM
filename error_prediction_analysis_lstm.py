# Bloc 5: Analyse des Erreurs de Prédiction
# Ce bloc suppose que les imports et certaines fonctions utilitaires (comme get_second_click_time)
# ont été définis dans des blocs précédents si exécuté dans un notebook.

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paramètres pour l'Analyse des Erreurs (doivent être cohérents avec l'entraînement) ---
WINDOW_SIZE = 150  # Doit correspondre au WINDOW_SIZE de l'entraînement du modèle à analyser
PREDICTION_HORIZON = 25 # Utilisé par create_sequences_and_labels_for_file pour le mapping
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

# Répertoire contenant les données de "training" qui serviront de test pour cette analyse
TRAINING_DATA_DIR_FOR_ANALYSIS = "training_dataset"

# Critères pour identifier une prédiction de déblocage par le modèle
PREDICTION_THRESHOLD_ANALYSIS = 0.7
PERSISTENCE_STEPS_ANALYSIS = int(0.1 * SAMPLING_RATE) # ex: 0.1s

# Fenêtres de tolérance pour catégoriser les prédictions
TOLERANCE_EARLY_S_ANALYSIS = 0.3
TOLERANCE_LATE_S_ANALYSIS = 0.7
MAX_ACCEPTABLE_DELAY_S_ANALYSIS = 2.0

# --- Fonctions Utilitaires (récupérées ou adaptées) ---
DEBUG_FILE_COUNT_LOAD_PROCESS_ANALYSIS = 0 # Compteur de débogage spécifique à cette fonction
MAX_DEBUG_FILES_LOAD_PROCESS_ANALYSIS = 2

def get_second_click_time_analysis(button_states, sampling_rate_local):
    # Identique à la fonction get_second_click_time du script principal
    clicks_start_indices = np.where(np.diff(button_states.astype(int)) == 1)[0] + 1
    if len(button_states) > 0 and button_states[0] == 1:
        if not (len(clicks_start_indices) > 0 and clicks_start_indices[0] == 0):
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)
        elif not np.any(clicks_start_indices == 0): # S'assurer de ne pas dupliquer si déjà présent
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)
    if len(clicks_start_indices) >= 2:
        return clicks_start_indices[1], clicks_start_indices[1] / sampling_rate_local
    return None, None

def create_sequences_and_labels_for_file_analysis(df, window_size, prediction_horizon, features, mode="training", sampling_rate_local=50, filename_for_debug="Unknown"):
    # Identique à la fonction create_sequences_and_labels_for_file du script principal
    # mais on utilise un compteur de debug séparé pour ne pas interférer
    global DEBUG_FILE_COUNT_LOAD_PROCESS_ANALYSIS
    X, y = [], [] # y ne sera pas utilisé pour l'analyse de timing directement, mais la fonction le retourne
    raw_button_state = df['Button_State'].values.astype(int)
    actual_second_click_idx, actual_second_click_time = None, None

    if DEBUG_FILE_COUNT_LOAD_PROCESS_ANALYSIS < MAX_DEBUG_FILES_LOAD_PROCESS_ANALYSIS and mode == "training":
        print(f"\n[Debug create_sequences_analysis] Fichier: {filename_for_debug}, Mode: {mode}")
        s_idx_dbg, s_time_dbg = get_second_click_time_analysis(raw_button_state, sampling_rate_local)
        if s_idx_dbg is not None: print(f"[Debug create_sequences_analysis] 2e clic trouvé à l'index {s_idx_dbg} (temps {s_time_dbg:.2f}s)")
        else: print(f"[Debug create_sequences_analysis] PAS de 2e clic trouvé.")

    if mode == "training": # On a besoin du temps du 2e clic pour l'analyse
        actual_second_click_idx, actual_second_click_time = get_second_click_time_analysis(raw_button_state, sampling_rate_local)
        if actual_second_click_idx is None and DEBUG_FILE_COUNT_LOAD_PROCESS_ANALYSIS < MAX_DEBUG_FILES_LOAD_PROCESS_ANALYSIS:
            print(f"Avertissement [Fichier: {filename_for_debug}]: Aucun 2e clic pour étiquetage (analyse).")

    data_for_sequences = df[features].copy()
    positive_labels_found_in_file = 0 # Juste pour le debug
    for i in range(len(data_for_sequences) - window_size - prediction_horizon + 1):
        input_seq = data_for_sequences.iloc[i : i + window_size].values
        label = 0 # L'étiquette réelle de la fenêtre n'est pas le focus principal de cette analyse de timing
        if mode == "training" and actual_second_click_idx is not None:
            start_predict_window_idx = i + window_size
            end_predict_window_idx = i + window_size + prediction_horizon
            if start_predict_window_idx <= actual_second_click_idx < end_predict_window_idx:
                label = 1
                positive_labels_found_in_file +=1
        X.append(input_seq)
        y.append(label) # On retourne y pour la cohérence, mais on utilisera surtout actual_second_click_time

    if DEBUG_FILE_COUNT_LOAD_PROCESS_ANALYSIS < MAX_DEBUG_FILES_LOAD_PROCESS_ANALYSIS and mode == "training":
        print(f"[Debug create_sequences_analysis] Fichier {filename_for_debug}: {positive_labels_found_in_file} étiquettes positives (1) théoriques créées.")
        DEBUG_FILE_COUNT_LOAD_PROCESS_ANALYSIS += 1
    return np.array(X), np.array(y), actual_second_click_time


def load_data_for_error_analysis(main_directory, window_size, prediction_horizon, features, sampling_rate_local=50, scaler_obj=None):
    # Similaire à load_process_all_simulations mais adapté pour l'analyse
    all_X_scaled_for_file = {} # Dictionnaire pour stocker X par fichier
    all_mapping_info_for_file = {} # Dictionnaire pour stocker les infos de mapping par fichier

    print(f"\nAnalyse du répertoire (pour erreurs): {os.path.abspath(main_directory)}")
    if not os.path.isdir(main_directory):
        print(f"Erreur : Le répertoire '{main_directory}' n'existe pas."); return {}, {}

    # On cible les fichiers _training car ils ont le 2e clic
    simulation_files = [f for f in os.listdir(main_directory)
                        if (f.endswith('_training.xlsx') or (f.endswith('.xlsx') and 'inference' not in f.lower() and 'full_for_plot' not in f.lower()))
                        and os.path.isfile(os.path.join(main_directory, f))]

    print(f"Fichiers trouvés pour l'analyse d'erreur: {simulation_files if simulation_files else 'Aucun'}")
    if not simulation_files: return {}, {}

    file_count = 0
    for filename in simulation_files:
        filepath = os.path.join(main_directory, filename)
        current_file_X_list = []
        current_file_mapping_list = []
        try:
            df = pd.read_excel(filepath)
            file_count += 1
            if not all(feat in df.columns for feat in features):
                print(f"Avertissement: Caractéristiques manquantes dans {filename}. Ignoré."); continue
            if 'Button_State' not in df.columns:
                print(f"Avertissement: 'Button_State' manquant dans {filename}. Ignoré."); continue

            # On utilise create_sequences_and_labels_for_file_analysis pour obtenir le temps du clic réel
            # Les étiquettes y_file ne sont pas primordiales ici, mais X_file et second_click_time_file le sont.
            X_file, _, second_click_time_file = create_sequences_and_labels_for_file_analysis(
                df, window_size, prediction_horizon, features, "training", sampling_rate_local, filename_for_debug=filename)

            if X_file.shape[0] > 0:
                # Normaliser X_file avec le scaler fourni
                if scaler_obj:
                    num_s, num_ts, num_f = X_file.shape
                    X_file_reshaped = X_file.reshape(-1, num_f)
                    X_file_scaled_reshaped = scaler_obj.transform(X_file_reshaped)
                    X_file_scaled = X_file_scaled_reshaped.reshape(num_s, num_ts, num_f)
                    all_X_scaled_for_file[filename] = X_file_scaled
                else:
                    print(f"Avertissement: Scaler non fourni pour {filename}. Les données ne seront pas normalisées.")
                    all_X_scaled_for_file[filename] = X_file # Utiliser non normalisé si pas de scaler

                # Stocker les infos de mapping pour ce fichier
                for seq_idx in range(X_file.shape[0]):
                    current_file_mapping_list.append({
                        "filename": filename,
                        "sequence_index_in_file": seq_idx,
                        "prediction_window_start_time": (seq_idx + window_size) / sampling_rate_local,
                        "actual_second_click_time": second_click_time_file
                    })
                all_mapping_info_for_file[filename] = current_file_mapping_list
            else:
                print(f"Aucune séquence générée pour {filename}")

        except Exception as e: print(f"Erreur traitant {filename} pour l'analyse: {e}")

    print(f"{file_count} fichiers traités pour l'analyse des erreurs.")
    return all_X_scaled_for_file, all_mapping_info_for_file


def find_predicted_deblocage_event_time_analysis(probabilities, window_start_times, threshold, persistence_steps):
    if probabilities is None or len(probabilities) < persistence_steps: return None
    for i in range(len(probabilities) - persistence_steps + 1):
        if np.all(probabilities[i : i + persistence_steps] > threshold):
            return window_start_times[i]
    return None

# --- Début du Script d'Analyse des Erreurs ---
print("\n--- DÉBUT DE L'ANALYSE DES ERREURS DE PRÉDICTION ---")

# Demander quel modèle analyser
model_type_to_analyze_errors = ""
while model_type_to_analyze_errors not in ["lstm", "cnn-lstm"]:
    model_type_to_analyze_errors = input("Quel type de modèle entraîné souhaitez-vous analyser pour les erreurs ('lstm' ou 'cnn-lstm') ? ").lower()
print(f"Choix pour l'analyse des erreurs : Modèle {model_type_to_analyze_errors.upper()}")

model_filepath_for_errors = f"{BASE_MODEL_FILENAME}_{model_type_to_analyze_errors}.h5"
scaler_filepath_for_errors = f"{BASE_SCALER_FILENAME}_{model_type_to_analyze_errors}.joblib"

if not (os.path.exists(model_filepath_for_errors) and os.path.exists(scaler_filepath_for_errors)):
    print(f"Modèle ({model_filepath_for_errors}) ou Scaler ({scaler_filepath_for_errors}) non trouvé.")
    print("Assurez-vous d'avoir entraîné ce type de modèle et que les fichiers sont présents.")
else:
    model_for_error_analysis = load_model(model_filepath_for_errors)
    scaler_for_error_analysis = joblib.load(scaler_filepath_for_errors)
    print(f"Modèle et scaler chargés pour l'analyse des erreurs ({model_type_to_analyze_errors}).")

    # Réinitialiser le compteur de debug avant de charger les données pour l'analyse
    DEBUG_FILE_COUNT_LOAD_PROCESS_ANALYSIS = 0

    # Charger les données de test (qui sont les données d'entraînement car elles ont le 2e clic)
    # La fonction retourne un dictionnaire de DataFrames X normalisés, et un dictionnaire de mappings
    dict_X_test_analysis, dict_mapping_info_analysis = load_data_for_error_analysis(
        TRAINING_DATA_DIR_FOR_ANALYSIS,
        window_size=WINDOW_SIZE, # Utiliser WINDOW_SIZE défini globalement (150)
        prediction_horizon=PREDICTION_HORIZON, # Nécessaire pour create_sequences...
        features=FEATURES_TO_USE,
        sampling_rate_local=SAMPLING_RATE,
        scaler_obj=scaler_for_error_analysis # Utiliser le scaler déjà fitté
    )

    if dict_X_test_analysis:
        print(f"\nDonnées de {len(dict_X_test_analysis)} fichiers chargées pour l'analyse des erreurs.")

        time_differences_list = []
        prediction_categories_dict = {"hit": 0, "early": 0, "late": 0, "missed_event": 0, "no_true_event_in_file": 0}

        for filename, X_file_scaled in dict_X_test_analysis.items():
            if X_file_scaled.shape[0] == 0:
                continue

            file_mapping_info = dict_mapping_info_analysis.get(filename, [])
            if not file_mapping_info:
                print(f"Pas d'info de mapping pour {filename}, impossible de trouver le temps de clic réel.")
                prediction_categories_dict["no_true_event_in_file"] += 1
                continue

            # Le temps de clic réel est le même pour toutes les séquences d'un même fichier
            actual_click_time_for_file = file_mapping_info[0]["actual_second_click_time"]

            if actual_click_time_for_file is None:
                prediction_categories_dict["no_true_event_in_file"] += 1
                continue

            probabilities_file = model_for_error_analysis.predict(X_file_scaled).flatten()

            # Les temps de début de fenêtre de prédiction pour ce fichier
            file_prediction_window_start_times = [m["prediction_window_start_time"] for m in file_mapping_info]

            predicted_time = find_predicted_deblocage_event_time_analysis(
                probabilities_file, file_prediction_window_start_times,
                PREDICTION_THRESHOLD_ANALYSIS, PERSISTENCE_STEPS_ANALYSIS
            )

            if predicted_time is not None:
                diff = predicted_time - actual_click_time_for_file
                time_differences_list.append(diff)
                if diff < (-TOLERANCE_EARLY_S_ANALYSIS): prediction_categories_dict["early"] += 1
                elif diff > TOLERANCE_LATE_S_ANALYSIS:
                    if diff <= MAX_ACCEPTABLE_DELAY_S_ANALYSIS: prediction_categories_dict["late"] += 1
                    else: prediction_categories_dict["missed_event"] +=1
                else: prediction_categories_dict["hit"] += 1
            else:
                prediction_categories_dict["missed_event"] += 1

        if time_differences_list:
            time_differences_array = np.array(time_differences_list)
            plt.figure(figsize=(12, 7))
            sns.histplot(time_differences_array, kde=True, bins=40)
            plt.title(f"Distribution des Écarts Temporels de Prédiction ({model_type_to_analyze_errors.upper()})")
            plt.xlabel("Écart de Temps (s) [Prédit - Réel] (Négatif = Trop Tôt)")
            plt.ylabel("Nombre de Simulations")
            plt.grid(True, linestyle='--')
            plt.axvline(-TOLERANCE_EARLY_S_ANALYSIS, color='orange', linestyle='--', label=f'Limite Trop Tôt (-{TOLERANCE_EARLY_S_ANALYSIS}s)')
            plt.axvline(TOLERANCE_LATE_S_ANALYSIS, color='red', linestyle='--', label=f'Limite Trop Tard (+{TOLERANCE_LATE_S_ANALYSIS}s)')
            plt.axvline(0, color='green', linestyle='-', label='Prédiction Parfaite')
            plt.legend()
            plt.show()

            print(f"\nStatistiques sur les Écarts Temporels ({model_type_to_analyze_errors.upper()}) :")
            print(f"  Nombre total de simulations avec 2e clic analysées: {len(time_differences_array)}")
            print(f"  Moyenne de l'écart: {np.mean(time_differences_array):.3f} s")
            print(f"  Médiane de l'écart: {np.median(time_differences_array):.3f} s")
            print(f"  Écart-type de l'écart: {np.std(time_differences_array):.3f} s")
            print(f"  Min écart: {np.min(time_differences_array):.3f} s (Prédiction la plus précoce par rapport au réel)")
            print(f"  Max écart: {np.max(time_differences_array):.3f} s (Prédiction la plus tardive par rapport au réel)")

            print(f"\nCatégorisation des Prédictions ({model_type_to_analyze_errors.upper()}) :")
            total_evaluated_events_for_cat = sum(prediction_categories_dict[k] for k in ["hit", "early", "late", "missed_event"])
            if total_evaluated_events_for_cat > 0:
                for cat, count in prediction_categories_dict.items():
                    if cat != "no_true_event_in_file":
                        print(f"  {cat.capitalize()}: {count} ({count/total_evaluated_events_for_cat*100:.1f}%)")
            if prediction_categories_dict["no_true_event_in_file"] > 0:
                print(f"  Fichiers sans 2e clic réel (non analysés pour le timing): {prediction_categories_dict['no_true_event_in_file']}")
        else:
            print("Aucun écart temporel n'a pu être calculé. Cela peut arriver si aucun événement n'a été prédit ou si aucun fichier de test ne contenait de 2e clic.")
    else:
        print(f"Aucune donnée de test n'a pu être chargée depuis {TRAINING_DATA_DIR_FOR_ANALYSIS} pour l'analyse des erreurs.")

