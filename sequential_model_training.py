# Bloc 1: Imports et Paramètres Globaux pour l'IA
import matplotlib.pyplot as plt 
import numpy as np
import os, pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import joblib


SAMPLING_RATE = 50
DT = 1.0 / SAMPLING_RATE
WINDOW_SIZE = 150 # nb de points à prendre (freq=50Hz)
PREDICTION_HORIZON = 25
FEATURES_TO_USE = [
    'Acc_X_Raw (m/s^2)', 'Acc_Y_Raw (m/s^2)', 'Acc_Z_Raw (m/s^2)',
    'Gyro_X (rad/s)', 'Gyro_Y (rad/s)', 'Gyro_Z (rad/s)',
    'Force_Cable1 (N)', 'Force_Cable2 (N)',
    'Vel_X_Noisy (m/s)', 'Vel_Y_Noisy (m/s)', 'Vel_Z_Noisy (m/s)',
    'Angle_Pitch_Noisy (rad)', 'Angle_Yaw_Noisy (rad)'
]
SAMPLING_RATE = 50

# --- Bloc 2: Fonctions de Chargement et de Prétraitement des Données ---

# Compteur global pour limiter les messages de débogage aux premiers fichiers
DEBUG_FILE_COUNT_CREATE_SEQ = 0
MAX_DEBUG_FILES_CREATE_SEQ = 3

def get_second_click_time(button_states, sampling_rate_local):
    """Trouve l'indice et le temps du début du deuxième clic de bouton."""
    # Un clic est marqué par une transition de 0 à 1
    # np.diff() calcule la différence entre éléments consécutifs.
    # Une transition 0 -> 1 donnera une différence de 1.
    # L'indice retourné par np.where est celui *avant* la transition, donc on ajoute 1.
    clicks_start_indices = np.where(np.diff(button_states) == 1)[0] + 1

    # Gérer le cas où le premier clic est au tout début (index 0)
    if button_states[0] == 1:
        # S'il y avait déjà un clic détecté par diff, et que c'est le même, on ne le rajoute pas
        if not (len(clicks_start_indices) > 0 and clicks_start_indices[0] == 0):
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)
        elif not np.any(clicks_start_indices == 0): # S'assurer de ne pas dupliquer si déjà présent
             clicks_start_indices = np.insert(clicks_start_indices, 0, 0)


    if len(clicks_start_indices) >= 2:
        second_click_start_idx = clicks_start_indices[1]
        return second_click_start_idx, second_click_start_idx / sampling_rate_local
    return None, None

def create_sequences_and_labels_for_file(df, window_size, prediction_horizon, features, mode="training", sampling_rate_local=50, filename_for_debug="Unknown"):
    global DEBUG_FILE_COUNT_CREATE_SEQ

    X, y = [], []
    # S'assurer que Button_State est bien de type entier pour np.diff
    raw_button_state = df['Button_State'].values.astype(int)
    actual_second_click_idx, actual_second_click_time = None, None

    if DEBUG_FILE_COUNT_CREATE_SEQ < MAX_DEBUG_FILES_CREATE_SEQ:
        print(f"\n[Debug create_sequences] Fichier: {filename_for_debug}, Mode: {mode}")
        print(f"[Debug create_sequences] Button_State (premiers 200 points ou moins): {raw_button_state[:200]}")

        # Test de get_second_click_time pour ce fichier
        s_idx_dbg, s_time_dbg = get_second_click_time(raw_button_state, sampling_rate_local)
        if s_idx_dbg is not None:
            print(f"[Debug create_sequences] get_second_click_time a trouvé un 2e clic à l'index {s_idx_dbg} (temps {s_time_dbg:.2f}s)")
        else:
            print(f"[Debug create_sequences] get_second_click_time N'A PAS trouvé de 2e clic.")

    if mode == "training":
        actual_second_click_idx, actual_second_click_time = get_second_click_time(raw_button_state, sampling_rate_local)
        if actual_second_click_idx is None:
            if DEBUG_FILE_COUNT_CREATE_SEQ < MAX_DEBUG_FILES_CREATE_SEQ: # Limiter l'affichage des avertissements
                print(f"Avertissement [Fichier: {filename_for_debug}]: Aucun second clic trouvé pour l'étiquetage. Toutes les étiquettes pour ce fichier seront 0.")

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

    if DEBUG_FILE_COUNT_CREATE_SEQ < MAX_DEBUG_FILES_CREATE_SEQ:
        print(f"[Debug create_sequences] Fichier {filename_for_debug}: {positive_labels_found_in_file} étiquettes positives (1) créées.")
        if mode == "training": DEBUG_FILE_COUNT_CREATE_SEQ += 1

    return np.array(X), np.array(y), actual_second_click_time

def load_process_all_simulations(main_directory, mode, window_size, prediction_horizon, features, sampling_rate_local=50, scaler_obj=None):
    all_X = []
    all_y = []
    all_original_indices_and_files = []

    print(f"\nAnalyse du répertoire : {os.path.abspath(main_directory)} pour le mode '{mode}'")
    if not os.path.isdir(main_directory):
        print(f"Erreur : Le répertoire '{main_directory}' n'existe pas.")
        return np.array([]), np.array([]), [], None

    if mode == "training":
        simulation_files = [f for f in os.listdir(main_directory)
                            if (f.endswith('_training.xlsx') or (f.endswith('.xlsx') and 'inference' not in f.lower() and 'full_for_plot' not in f.lower()))
                            and os.path.isfile(os.path.join(main_directory, f))]
    elif mode == "inference":
        simulation_files = [f for f in os.listdir(main_directory)
                            if f.endswith('_inference.xlsx')
                            and os.path.isfile(os.path.join(main_directory, f))]
    else:
        simulation_files = []

    print(f"Fichiers trouvés correspondant au mode '{mode}': {simulation_files if simulation_files else 'Aucun'}")
    if not simulation_files:
        return np.array([]), np.array([]), [], None

    file_count = 0
    for filename in simulation_files:
        filepath = os.path.join(main_directory, filename)
        try:
            df = pd.read_excel(filepath)
            file_count += 1
            # Vérifier la présence de toutes les colonnes AVANT d'appeler create_sequences
            if not all(feat in df.columns for feat in features):
                print(f"Avertissement: Certaines caractéristiques de FEATURES_TO_USE sont manquantes dans {filename}. Fichier ignoré.")
                continue
            if 'Button_State' not in df.columns:
                print(f"Avertissement: Colonne 'Button_State' manquante dans {filename}. Fichier ignoré.")
                continue

            X_file, y_file, second_click_time_file = create_sequences_and_labels_for_file(
                df, window_size, prediction_horizon, features, mode, sampling_rate_local, filename_for_debug=filename
            )

            if X_file.shape[0] > 0:
                all_X.append(X_file)
                if mode == "training": # Les étiquettes ne sont pertinentes que pour le training
                    all_y.append(y_file)

                for seq_idx in range(X_file.shape[0]):
                    prediction_window_start_time = (seq_idx + window_size) / sampling_rate_local
                    all_original_indices_and_files.append({
                        "filename": filename,
                        "sequence_index_in_file": seq_idx,
                        "prediction_window_start_time": prediction_window_start_time,
                        "actual_second_click_time": second_click_time_file if mode == "training" else None
                    })
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {filename}: {e}")

    print(f"{file_count} fichiers traités dans {main_directory}.")
    if not all_X:
        print("Aucune séquence n'a été créée à partir des fichiers trouvés.")
        return np.array([]), np.array([]), [], None

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0) if mode == "training" and all_y else np.array([])

    if X_combined.shape[0] == 0: # Vérification supplémentaire
        print("X_combined est vide après concaténation.")
        return np.array([]), np.array([]), [], None

    num_samples, num_timesteps, num_features = X_combined.shape
    X_reshaped = X_combined.reshape(-1, num_features)

    current_scaler = scaler_obj
    if mode == "training":
        print("Ajustement du scaler sur les données d'entraînement...")
        current_scaler = StandardScaler()
        X_scaled_reshaped = current_scaler.fit_transform(X_reshaped)
        print("Scaler ajusté.")
    elif current_scaler is not None:
        print("Utilisation du scaler fourni pour transformer les données...")
        X_scaled_reshaped = current_scaler.transform(X_reshaped)
        print("Données transformées.")
    else:
        print("Avertissement: Mode inférence sans scaler pré-ajusté fourni. Les données ne seront pas normalisées.")
        X_scaled_reshaped = X_reshaped

    X_scaled = X_scaled_reshaped.reshape(num_samples, num_timesteps, num_features)

    return X_scaled, y_combined, all_original_indices_and_files, current_scaler


# --- Bloc 3: Construction du Modèle LSTM ---
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# --- Bloc 4: Entraînement du Modèle ---
TRAINING_DATA_DIR = "training_dataset"
MODEL_FILEPATH = "deblocage_lstm_model.h5"
SCALER_FILEPATH = "scaler.joblib"

print("\n--- Phase d'Entraînement ---")
# Réinitialiser le compteur de débogage avant de charger les données d'entraînement
DEBUG_FILE_COUNT_CREATE_SEQ = 0
X_train_full, y_train_full, train_mapping_info, scaler = load_process_all_simulations(
    TRAINING_DATA_DIR,
    mode="training",
    window_size=WINDOW_SIZE,
    prediction_horizon=PREDICTION_HORIZON,
    features=FEATURES_TO_USE,
    sampling_rate_local=SAMPLING_RATE
)

if X_train_full.shape[0] > 0 and y_train_full.shape[0] > 0: # S'assurer que y_train_full n'est pas vide
    joblib.dump(scaler, SCALER_FILEPATH)
    print(f"Scaler sauvegardé dans {SCALER_FILEPATH}")

    # Vérifier s'il y a au moins deux classes pour stratify
    can_stratify = len(np.unique(y_train_full)) > 1

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, shuffle=True,
        stratify=y_train_full if can_stratify else None
    )

    print(f"Taille des données d'entraînement: {X_train.shape}, Étiquettes: {y_train.shape}")
    print(f"Taille des données de validation: {X_val.shape}, Étiquettes: {y_val.shape}")
    if len(y_train) > 0: print(f"Distribution des étiquettes (entraînement): {np.bincount(y_train.astype(int))}") # Cast en int pour bincount
    if len(y_val) > 0: print(f"Distribution des étiquettes (validation): {np.bincount(y_val.astype(int))}")

    # Vérifier la présence d'exemples positifs
    has_positive_train = len(y_train) > 0 and np.sum(y_train) > 0
    has_positive_val = len(y_val) > 0 and np.sum(y_val) > 0

    if not has_positive_train:
        print("Avertissement: L'ensemble d'entraînement ne contient pas d'exemples positifs (étiquette 1).")
    if not has_positive_val and X_val.shape[0] > 0 : # X_val peut être vide si X_train_full est trop petit
         print("Avertissement: L'ensemble de validation ne contient pas d'exemples positifs (étiquette 1).")

    if X_train.shape[0] > 0 and X_val.shape[0] > 0 and has_positive_train :
        model = build_lstm_model(input_shape=(WINDOW_SIZE, X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(MODEL_FILEPATH, monitor='val_accuracy', save_best_only=True, verbose=1)

        count_0 = np.sum(y_train == 0)
        count_1 = np.sum(y_train == 1) # Sera 0 si has_positive_train est faux, mais on a la condition avant

        # S'assurer que count_1 n'est pas zéro avant de calculer le poids
        if count_1 == 0:
            print("Erreur critique: Aucun exemple positif dans l'ensemble d'entraînement, impossible de calculer class_weight pour la classe 1.")
            class_w = {0: 1.0, 1: 1.0} # Poids par défaut
        else:
            weight_for_0 = (1 / count_0) * (len(y_train) / 2.0) if count_0 > 0 else 1.0
            weight_for_1 = (1 / count_1) * (len(y_train) / 2.0)
            class_w = {0: weight_for_0, 1: weight_for_1}
        print(f"Poids des classes calculés: {class_w}")

        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            class_weight=class_w
        )
        print(f"Modèle entraîné et meilleur modèle sauvegardé dans {MODEL_FILEPATH}")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Perte (train)'); plt.plot(history.history['val_loss'], label='Perte (val)'); plt.title('Perte'); plt.xlabel('Époque'); plt.ylabel('Perte'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2); plt.plot(history.history['accuracy'], label='Précision (train)'); plt.plot(history.history['val_accuracy'], label='Précision (val)'); plt.title('Précision'); plt.xlabel('Époque'); plt.ylabel('Précision'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.show()
    else:
        print("Pas assez de données, pas d'exemples positifs dans l'entraînement, ou ensemble de validation vide. L'entraînement est annulé.")
else:
    print("Aucune donnée d'entraînement n'a été chargée ou traitée. Vérifiez TRAINING_DATA_DIR et les fichiers.")

# --- Bloc 5: Évaluation du Modèle ---
print("\n--- Phase d'Évaluation (sur les données de validation) ---")
if os.path.exists(MODEL_FILEPATH) and 'X_val' in locals() and X_val.shape[0] > 0 and 'y_val' in locals() and y_val.shape[0] > 0:
    model = load_model(MODEL_FILEPATH)
    y_pred_proba_val = model.predict(X_val)
    y_pred_val = (y_pred_proba_val > 0.5).astype(int).flatten()

    print("\nRapport de classification (sur les fenêtres de validation):")
    # S'assurer que y_val et y_pred_val ont des labels pour les deux classes si possible, sinon le rapport peut être limité
    target_names_report = ['Pas de Clic (Pred)', 'Clic (Pred)']
    # Vérifier si les deux classes sont présentes dans y_val ou y_pred_val pour éviter les warnings de sklearn
    unique_labels_true = np.unique(y_val)
    unique_labels_pred = np.unique(y_pred_val)

    if len(unique_labels_true) < 2 or len(unique_labels_pred) < 2:
        print("Avertissement: Toutes les étiquettes (vraies ou prédites) appartiennent à une seule classe en validation.")
        # Afficher le rapport avec les labels présents
        present_labels = np.union1d(unique_labels_true, unique_labels_pred).astype(int)
        custom_target_names = [target_names_report[i] for i in present_labels]
        if not custom_target_names: # Si tout est vide
             print(classification_report(y_val, y_pred_val, zero_division=0))
        else:
             print(classification_report(y_val, y_pred_val, labels=present_labels, target_names=custom_target_names, zero_division=0))

    else:
        print(classification_report(y_val, y_pred_val, target_names=target_names_report, zero_division=0))

    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pas de Clic', 'Clic'], yticklabels=['Pas de Clic', 'Clic'])
    plt.title('Matrice de Confusion (Validation Windows)'); plt.ylabel('Vraie Étiquette'); plt.xlabel('Étiquette Prédite'); plt.show()

    print("\nL'évaluation du timing fin nécessite un post-traitement plus avancé.")
else:
    print("Modèle non entraîné ou pas de données de validation pour l'évaluation.")

# --- Bloc 6: Inférence sur de Nouvelles Données ---
INFERENCE_DATA_DIR = "inference_dataset"

print("\n--- Phase d'Inférence ---")
DEBUG_FILE_COUNT_CREATE_SEQ = 0 # Réinitialiser pour l'inférence
if os.path.exists(MODEL_FILEPATH) and os.path.exists(SCALER_FILEPATH):
    inference_model = load_model(MODEL_FILEPATH)
    scaler_for_inference = joblib.load(SCALER_FILEPATH)
    print(f"Modèle chargé depuis {MODEL_FILEPATH} et scaler depuis {SCALER_FILEPATH} pour l'inférence.")

    X_inference, _, inference_mapping_info, _ = load_process_all_simulations(
        INFERENCE_DATA_DIR,
        mode="inference",
        window_size=WINDOW_SIZE,
        prediction_horizon=PREDICTION_HORIZON,
        features=FEATURES_TO_USE,
        sampling_rate_local=SAMPLING_RATE,
        scaler_obj=scaler_for_inference
    )

    if X_inference.shape[0] > 0:
        print(f"Données d'inférence chargées et prétraitées: {X_inference.shape}")
        predictions_proba_inference = inference_model.predict(X_inference)

        first_inference_file_name = None
        if inference_mapping_info: # S'assurer que la liste n'est pas vide
            first_inference_file_name = inference_mapping_info[0]["filename"]

        indices_for_first_file = [i for i, mapping in enumerate(inference_mapping_info) if mapping["filename"] == first_inference_file_name]

        if indices_for_first_file and first_inference_file_name is not None: # Vérifier aussi que le nom du fichier est défini
            start_idx = indices_for_first_file[0]
            end_idx = indices_for_first_file[-1] + 1

            plt.figure(figsize=(15, 6))
            prediction_times_for_plot = [m["prediction_window_start_time"] for m in inference_mapping_info[start_idx:end_idx]]

            plt.plot(prediction_times_for_plot, predictions_proba_inference[start_idx:end_idx], label=f"Probabilité de déblocage (Sim: {first_inference_file_name})")
            plt.axhline(0.5, color='r', linestyle='--', label="Seuil de décision (0.5)")
            plt.title(f"Probabilités de Prédiction sur la Simulation d'Inférence: {first_inference_file_name}")
            plt.xlabel("Temps (s) - Début de la fenêtre de prédiction")
            plt.ylabel("Probabilité de Déblocage"); plt.legend(); plt.grid(True); plt.ylim(0,1); plt.show()

            threshold = 0.7
            persistence_steps = int(0.1 * SAMPLING_RATE) # Ex: 0.1s de persistance
            predicted_deblocage_time = None

            proba_first_file = predictions_proba_inference[start_idx:end_idx].flatten()
            for k in range(len(proba_first_file) - persistence_steps + 1):
                if np.all(proba_first_file[k : k + persistence_steps] > threshold):
                    predicted_deblocage_time = prediction_times_for_plot[k]
                    print(f"Événement de déblocage PRÉDIT pour {first_inference_file_name} à ~{predicted_deblocage_time:.2f}s (prob > {threshold} pendant {persistence_steps*DT:.2f}s)")
                    break
            if predicted_deblocage_time is None:
                print(f"Aucun événement de déblocage prédit pour {first_inference_file_name} avec les critères actuels.")
        else:
            print("Impossible d'isoler les prédictions pour la première simulation d'inférence ou aucune information de mapping.")

    else:
        print("Aucune donnée d'inférence n'a été chargée. Vérifiez INFERENCE_DATA_DIR et les fichiers.")
else:
    print(f"Aucun modèle ({MODEL_FILEPATH}) ou scaler ({SCALER_FILEPATH}) entraîné trouvé. Veuillez d'abord entraîner le modèle.")

