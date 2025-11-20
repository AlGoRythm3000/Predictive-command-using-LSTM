# Bloc (Cellule) pour l'Analyse des Erreurs de Prédiction
# Ce bloc suppose que les blocs précédents (imports, fonctions utilitaires,
# et potentiellement l'entraînement si vous voulez ré-évaluer sur les données de validation)
# ont été exécutés ou que les artefacts nécessaires (modèle, scaler) sont disponibles.

from tensorflow.keras.models import load_model # Pour charger le modèle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import numpy as np

import pandas as pd
import seaborn as sns
import os
import joblib

# --- Paramètres pour l'Analyse des Erreurs ---
# Assurez-vous que ces paramètres sont cohérents avec ceux utilisés pour l'entraînement
WINDOW_SIZE_ANALYSIS = 150  # CORRIGÉ: Doit correspondre au WINDOW_SIZE de l'entraînement (3 secondes * 50 Hz)
PREDICTION_HORIZON_ANALYSIS = 25 # Utilisé par create_sequences_and_labels_for_file (si on l'utilisait)
FEATURES_TO_USE_ANALYSIS = [
    'Acc_X_Raw (m/s^2)', 'Acc_Y_Raw (m/s^2)', 'Acc_Z_Raw (m/s^2)',
    'Gyro_X (rad/s)', 'Gyro_Y (rad/s)', 'Gyro_Z (rad/s)',
    'Force_Cable1 (N)', 'Force_Cable2 (N)',
    'Vel_X_Noisy (m/s)', 'Vel_Y_Noisy (m/s)', 'Vel_Z_Noisy (m/s)',
    'Angle_Pitch_Noisy (rad)', 'Angle_Yaw_Noisy (rad)'
]
SAMPLING_RATE_ANALYSIS = 50
DT_ANALYSIS = 1.0 / SAMPLING_RATE_ANALYSIS

# Chemins vers le modèle et le scaler sauvegardés
MODEL_FILEPATH_ANALYSIS = "deblocage_lstm_model.h5"
SCALER_FILEPATH_ANALYSIS = "scaler.joblib"

# Répertoire contenant les données de "training" qui serviront de test pour cette analyse
# car elles contiennent le 2ème clic (vérité terrain)
TEST_DATA_DIR_FOR_ERROR_ANALYSIS = "training_dataset"

# Critères pour identifier une prédiction de déblocage par le modèle
PREDICTION_THRESHOLD = 0.7
PERSISTENCE_STEPS = int(0.1 * SAMPLING_RATE_ANALYSIS)

# Fenêtres de tolérance pour catégoriser les prédictions
TOLERANCE_EARLY_S = 0.3
TOLERANCE_LATE_S = 0.7
MAX_ACCEPTABLE_DELAY_S = 2.0

DEBUG_FILE_COUNT_ANALYSIS = 0
MAX_DEBUG_FILES_ANALYSIS = 2

def get_second_click_time_analysis(button_states, sampling_rate_local):
    clicks_start_indices = np.where(np.diff(button_states.astype(int)) == 1)[0] + 1
    if len(button_states) > 0 and button_states[0] == 1: # Vérifier que button_states n'est pas vide
        if not (len(clicks_start_indices) > 0 and clicks_start_indices[0] == 0):
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)
        elif not np.any(clicks_start_indices == 0):
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)
    if len(clicks_start_indices) >= 2:
        return clicks_start_indices[1], clicks_start_indices[1] / sampling_rate_local
    return None, None

def create_sequences_for_prediction_analysis(df, window_size, features):
    X = []
    if not all(feat in df.columns for feat in features): # Vérifier que toutes les features sont là
        print(f"Avertissement: Colonnes manquantes dans le DataFrame pour create_sequences_for_prediction_analysis. Features attendues: {features}, Colonnes DataFrame: {df.columns.tolist()}")
        return np.array(X)

    data_for_sequences = df[features].copy()
    for i in range(len(data_for_sequences) - window_size + 1):
        input_seq = data_for_sequences.iloc[i : i + window_size].values
        X.append(input_seq)
    return np.array(X)

def find_predicted_deblocage_event_time(probabilities, window_start_times, threshold, persistence_steps, dt_local):
    if probabilities is None or len(probabilities) < persistence_steps:
        return None
    for i in range(len(probabilities) - persistence_steps + 1):
        if np.all(probabilities[i : i + persistence_steps] > threshold):
            return window_start_times[i]
    return None

# --- Script Principal d'Analyse des Erreurs ---

print("--- Analyse des Erreurs de Prédiction ---")

if not (os.path.exists(MODEL_FILEPATH_ANALYSIS) and os.path.exists(SCALER_FILEPATH_ANALYSIS)):
    print(f"Erreur: Modèle ({MODEL_FILEPATH_ANALYSIS}) ou Scaler ({SCALER_FILEPATH_ANALYSIS}) non trouvé.")
    print("Veuillez exécuter le script d'entraînement d'abord.")
else:
    model = load_model(MODEL_FILEPATH_ANALYSIS)
    scaler = joblib.load(SCALER_FILEPATH_ANALYSIS)
    print(f"Modèle chargé depuis {MODEL_FILEPATH_ANALYSIS}")
    print(f"Scaler chargé depuis {SCALER_FILEPATH_ANALYSIS}")

    if not os.path.isdir(TEST_DATA_DIR_FOR_ERROR_ANALYSIS):
        print(f"Erreur: Le répertoire de test '{TEST_DATA_DIR_FOR_ERROR_ANALYSIS}' n'existe pas.")
    else:
        test_simulation_files = [f for f in os.listdir(TEST_DATA_DIR_FOR_ERROR_ANALYSIS)
                                 if (f.endswith('_training.xlsx') or (f.endswith('.xlsx') and 'inference' not in f.lower() and 'full_for_plot' not in f.lower()))
                                 and os.path.isfile(os.path.join(TEST_DATA_DIR_FOR_ERROR_ANALYSIS, f))]

        print(f"Fichiers de test trouvés dans '{TEST_DATA_DIR_FOR_ERROR_ANALYSIS}': {len(test_simulation_files)}")

        time_differences = []
        prediction_categories = {"hit": 0, "early": 0, "late": 0, "missed_event": 0, "no_true_event": 0}
        processed_files_count = 0

        for filename in test_simulation_files:
            filepath = os.path.join(TEST_DATA_DIR_FOR_ERROR_ANALYSIS, filename)
            try:
                df_test = pd.read_excel(filepath)
                processed_files_count += 1

                if 'Button_State' not in df_test.columns:
                    print(f"Avertissement: Colonne 'Button_State' manquante dans {filename}. Fichier ignoré pour l'analyse de timing.")
                    prediction_categories["no_true_event"] +=1 # Ou une autre catégorie d'erreur
                    continue

                button_state_test = df_test['Button_State'].values.astype(int)
                actual_second_click_idx, actual_second_click_time = get_second_click_time_analysis(button_state_test, SAMPLING_RATE_ANALYSIS)

                if actual_second_click_time is None:
                    prediction_categories["no_true_event"] += 1
                    if DEBUG_FILE_COUNT_ANALYSIS < MAX_DEBUG_FILES_ANALYSIS:
                        print(f"[Debug Analyse] Fichier {filename}: Pas de 2ème clic réel trouvé. Ignoré pour l'analyse d'erreur de timing.")
                    continue # Passer au fichier suivant si pas de 2e clic réel

                X_file_test_raw = create_sequences_for_prediction_analysis(df_test, WINDOW_SIZE_ANALYSIS, FEATURES_TO_USE_ANALYSIS)

                if X_file_test_raw.shape[0] == 0:
                    if DEBUG_FILE_COUNT_ANALYSIS < MAX_DEBUG_FILES_ANALYSIS:
                        print(f"[Debug Analyse] Fichier {filename}: Aucune séquence générée (peut-être trop court pour WINDOW_SIZE_ANALYSIS={WINDOW_SIZE_ANALYSIS}). Ignoré.")
                    continue

                num_s, num_ts, num_f = X_file_test_raw.shape
                X_file_test_reshaped = X_file_test_raw.reshape(-1, num_f)
                X_file_test_scaled_reshaped = scaler.transform(X_file_test_reshaped)
                X_file_test_scaled = X_file_test_scaled_reshaped.reshape(num_s, num_ts, num_f)

                probabilities_file = model.predict(X_file_test_scaled).flatten()

                # Le temps de début de la *fenêtre de prédiction* est (index_sequence + window_size) * dt
                # car la prédiction concerne ce qui se passe *après* la fenêtre d'entrée.
                prediction_window_start_times_file = (np.arange(len(probabilities_file)) + WINDOW_SIZE_ANALYSIS) * DT_ANALYSIS

                predicted_deblocage_time = find_predicted_deblocage_event_time(
                    probabilities_file,
                    prediction_window_start_times_file,
                    PREDICTION_THRESHOLD,
                    PERSISTENCE_STEPS,
                    DT_ANALYSIS
                )

                if DEBUG_FILE_COUNT_ANALYSIS < MAX_DEBUG_FILES_ANALYSIS:
                    print(f"[Debug Analyse] Fichier {filename}: Clic réel à {actual_second_click_time:.2f}s.")
                    if predicted_deblocage_time is not None:
                        print(f"[Debug Analyse] Fichier {filename}: Clic prédit à {predicted_deblocage_time:.2f}s.")
                    else:
                        print(f"[Debug Analyse] Fichier {filename}: Aucun clic prédit par le modèle avec les critères actuels.")

                if predicted_deblocage_time is not None:
                    diff = predicted_deblocage_time - actual_second_click_time
                    time_differences.append(diff)

                    if diff < (-TOLERANCE_EARLY_S):
                        prediction_categories["early"] += 1
                    elif diff > TOLERANCE_LATE_S:
                        if diff <= MAX_ACCEPTABLE_DELAY_S :
                            prediction_categories["late"] += 1
                        else:
                            prediction_categories["missed_event"] +=1
                    else:
                        prediction_categories["hit"] += 1
                else:
                    prediction_categories["missed_event"] += 1

                if DEBUG_FILE_COUNT_ANALYSIS < MAX_DEBUG_FILES_ANALYSIS and processed_files_count <= MAX_DEBUG_FILES_ANALYSIS:
                    DEBUG_FILE_COUNT_ANALYSIS +=1

            except ValueError as ve: # Attraper spécifiquement les erreurs de shape
                print(f"Erreur de valeur (potentiellement shape) lors du traitement de {filename}: {ve}")
                print("Vérifiez que WINDOW_SIZE_ANALYSIS correspond à la taille de fenêtre de l'entraînement du modèle.")
            except Exception as e:
                print(f"Erreur générale lors du traitement du fichier de test {filename} pour l'analyse d'erreur: {e}")

        print(f"\n{processed_files_count} fichiers de test traités pour l'analyse des erreurs.")
        if time_differences:
            time_differences = np.array(time_differences)

            plt.figure(figsize=(10, 6))
            sns.histplot(time_differences, kde=True, bins=30)
            plt.title("Distribution des Écarts Temporels (Prédit - Réel)")
            plt.xlabel("Écart de Temps (s) [Négatif = Trop Tôt, Positif = Trop Tard]")
            plt.ylabel("Nombre de Simulations")
            plt.grid(True, linestyle='--')
            plt.axvline(-TOLERANCE_EARLY_S, color='orange', linestyle='--', label=f'Limite Trop Tôt (-{TOLERANCE_EARLY_S}s)')
            plt.axvline(TOLERANCE_LATE_S, color='red', linestyle='--', label=f'Limite Trop Tard (+{TOLERANCE_LATE_S}s)')
            plt.axvline(0, color='green', linestyle='-', label='Prédiction Parfaite')
            plt.legend()
            plt.show()

            print("\nStatistiques sur les Écarts Temporels :")
            print(f"  Moyenne de l'écart: {np.mean(time_differences):.3f} s")
            print(f"  Médiane de l'écart: {np.median(time_differences):.3f} s")
            print(f"  Écart-type de l'écart: {np.std(time_differences):.3f} s")
            print(f"  Min écart: {np.min(time_differences):.3f} s")
            print(f"  Max écart: {np.max(time_differences):.3f} s")

            print("\nCatégorisation des Prédictions :")
            total_evaluated_events = prediction_categories["hit"] + prediction_categories["early"] + prediction_categories["late"] + prediction_categories["missed_event"]
            if total_evaluated_events > 0:
                print(f"  Bonnes Prédictions (Hits) [-{TOLERANCE_EARLY_S}s, +{TOLERANCE_LATE_S}s]: {prediction_categories['hit']} ({prediction_categories['hit']/total_evaluated_events*100:.1f}%)")
                print(f"  Prédictions Trop Précoces (< -{TOLERANCE_EARLY_S}s): {prediction_categories['early']} ({prediction_categories['early']/total_evaluated_events*100:.1f}%)")
                print(f"  Prédictions Trop Tardives (> +{TOLERANCE_LATE_S}s et <= {MAX_ACCEPTABLE_DELAY_S}s): {prediction_categories['late']} ({prediction_categories['late']/total_evaluated_events*100:.1f}%)")
                print(f"  Prédictions Manquées/Très Tardives (> {MAX_ACCEPTABLE_DELAY_S}s ou non prédit): {prediction_categories['missed_event']} ({prediction_categories['missed_event']/total_evaluated_events*100:.1f}%)")
            else:
                print("Aucun événement n'a pu être évalué (vérifiez les données et les seuils).")
            if prediction_categories["no_true_event"] > 0:
                 print(f"  Simulations ignorées car pas de 2ème clic réel: {prediction_categories['no_true_event']}")
        else:
            print("Aucun écart temporel n'a pu être calculé. Vérifiez les données de test et les prédictions du modèle.")
