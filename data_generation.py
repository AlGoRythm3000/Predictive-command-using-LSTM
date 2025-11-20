import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import os
import openpyxl

# --- Bloc 1: Paramètres Globaux et Fonctions Utilitaires ---
SAMPLING_RATE = 50
DT = 1.0 / SAMPLING_RATE
CHARGE_KG = 15
FORCE_DUE_A_LA_CHARGE = CHARGE_KG * 9.81
G_ACCEL = 9.81

NOISE_STD_ACC_BASE = 0.50
NOISE_STD_GYRO_BASE = 0.08
NOISE_STD_FORCE_BASE = 4.5

BASE_DUREE_CLIC_BOUTON_AVG = 0.15
RAND_FACTOR_CLIC = 0.8

BASE_DUREE_REDRESSEMENT_CHARGE = 1.5
BASE_DUREE_ROTATION = 1.5
BASE_DUREE_MARCHE_AVEC_CHARGE = 4.0
BASE_DUREE_FLEXION_POSE = 1.2
BASE_DUREE_AVANT_CLIC_DEBLOCAGE = 0.25
BASE_DUREE_REDRESSEMENT_SANS_CHARGE = 1.2
BASE_DUREE_POST_FINAL = 2.5

SUM_BASE_CORE_DURATIONS_TRAINING = (BASE_DUREE_CLIC_BOUTON_AVG * 2) + \
                                   BASE_DUREE_REDRESSEMENT_CHARGE + \
                                   BASE_DUREE_ROTATION + \
                                   BASE_DUREE_MARCHE_AVEC_CHARGE + \
                                   BASE_DUREE_FLEXION_POSE + \
                                   BASE_DUREE_AVANT_CLIC_DEBLOCAGE + \
                                   BASE_DUREE_REDRESSEMENT_SANS_CHARGE

SUM_BASE_CORE_DURATIONS_INFERENCE = BASE_DUREE_CLIC_BOUTON_AVG + \
                                    BASE_DUREE_REDRESSEMENT_CHARGE + \
                                    BASE_DUREE_ROTATION + \
                                    BASE_DUREE_MARCHE_AVEC_CHARGE + \
                                    BASE_DUREE_FLEXION_POSE

DELTA_HAUTEUR_REDRESSEMENT1 = 0.7
ANGLE_ROTATION_YAW = np.deg2rad(90)
DISTANCE_MARCHE = 2.0
DELTA_HAUTEUR_FLEXION_POSE = -0.3
ANGLE_FLEXION_PITCH_POSE = np.deg2rad(25)
# La variable globale angle_redressement_pitch_final = -ANGLE_FLEXION_PITCH_POSE est correcte ici si Bloc 1 est exécuté

INITIAL_POS_X = 0.0
INITIAL_POS_Y = 0.0
INITIAL_POS_Z = 0.5
INITIAL_ANGLE_ROLL = 0.0
INITIAL_ANGLE_PITCH = np.deg2rad(15)
INITIAL_ANGLE_YAW = 0.0

MIN_TARGET_TOTAL_DURATION = 7.0
MAX_TARGET_TOTAL_DURATION = 11.0
PHASE_VAR_FACTOR = 0.1

def generate_pulse(data_array, total_samples_local, start_time, end_time, amplitude):
    start_idx = max(0, int(start_time * SAMPLING_RATE))
    end_idx = min(total_samples_local, int(end_time * SAMPLING_RATE))
    if start_idx < end_idx:
        data_array[start_idx:end_idx] = amplitude
    return data_array

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_sigmoid_ramp(data_array, total_samples_local, start_time, end_time, start_value, end_value, k=8):
    start_idx = max(0, int(start_time * SAMPLING_RATE))
    end_idx = min(total_samples_local, int(end_time * SAMPLING_RATE))
    if start_idx < end_idx:
        duration_points = end_idx - start_idx
        if duration_points == 0: return data_array
        x_lin = np.linspace(-k/2, k/2, duration_points)
        scaled_sigmoid = sigmoid(x_lin)
        data_array[start_idx:end_idx] = start_value + (end_value - start_value) * scaled_sigmoid
    return data_array

def apply_acceleration_for_displacement(acc_axis_data, total_samples_local, start_time, end_time, displacement, is_z_axis=False):
    start_idx = max(0, int(start_time * SAMPLING_RATE))
    end_idx = min(total_samples_local, int(end_time * SAMPLING_RATE))
    duration_total_move = end_time - start_time

    if duration_total_move <= DT or start_idx >= end_idx:
        return acc_axis_data

    V_peak = (2 * displacement) / duration_total_move
    mid_time = start_time + duration_total_move / 2
    mid_idx = max(start_idx, min(end_idx -1, int(mid_time * SAMPLING_RATE)))

    duration_phase1 = mid_time - start_time
    current_phase_end_idx = mid_idx
    if duration_phase1 > DT/2 and start_idx < current_phase_end_idx :
        amplitude_acc1 = (2 * V_peak) / duration_phase1
        num_points_phase1 = current_phase_end_idx - start_idx
        if num_points_phase1 == 0 : return acc_axis_data
        t_pulse1 = np.linspace(0, duration_phase1, num_points_phase1, endpoint=False)
        acc_pulse1 = (amplitude_acc1 / 2) * (1 - np.cos(2 * np.pi * t_pulse1 / duration_phase1))
        if is_z_axis:
            acc_axis_data[start_idx:current_phase_end_idx] += acc_pulse1
        else:
            acc_axis_data[start_idx:current_phase_end_idx] = acc_pulse1

    duration_phase2 = end_time - mid_time
    current_phase_start_idx = mid_idx
    if duration_phase2 > DT/2 and current_phase_start_idx < end_idx:
        amplitude_acc2 = (2 * -V_peak) / duration_phase2
        num_points_phase2 = end_idx - current_phase_start_idx
        if num_points_phase2 == 0 : return acc_axis_data
        t_pulse2 = np.linspace(0, duration_phase2, num_points_phase2, endpoint=False)
        acc_pulse2 = (amplitude_acc2 / 2) * (1 - np.cos(2 * np.pi * t_pulse2 / duration_phase2))
        if is_z_axis:
            acc_axis_data[current_phase_start_idx:end_idx] += acc_pulse2
        else:
            acc_axis_data[current_phase_start_idx:end_idx] = acc_pulse2
    return acc_axis_data

def apply_angular_velocity_for_rotation(gyro_axis_data, total_samples_local, start_time, end_time, total_angle_change):
    start_idx = max(0, int(start_time * SAMPLING_RATE))
    end_idx = min(total_samples_local, int(end_time * SAMPLING_RATE))
    duration_pulse = end_time - start_time

    if duration_pulse <= DT or start_idx >= end_idx:
        return gyro_axis_data

    amplitude_omega_pulse = (2 * total_angle_change) / duration_pulse
    num_points_pulse = end_idx - start_idx
    if num_points_pulse == 0: return gyro_axis_data
    t_pulse_overall = np.linspace(0, duration_pulse, num_points_pulse, endpoint=False)
    gyro_pulse_overall = (amplitude_omega_pulse / 2) * (1 - np.cos(2 * np.pi * t_pulse_overall / duration_pulse))
    gyro_axis_data[start_idx:end_idx] = gyro_pulse_overall
    return gyro_axis_data

# --- Bloc 2: Fonction de Génération pour une Simulation ---
def generate_single_simulation_data(simulation_id, generation_mode="training"):
    """Génère les données pour une seule simulation avec variabilité temporelle et bruit."""
    print(f"Génération de la simulation {simulation_id} (Mode: {generation_mode})...")

    target_total_duration_sim = np.random.uniform(MIN_TARGET_TOTAL_DURATION, MAX_TARGET_TOTAL_DURATION)

    sum_base_durations_for_mode = SUM_BASE_CORE_DURATIONS_TRAINING if generation_mode == "training" else SUM_BASE_CORE_DURATIONS_INFERENCE

    target_core_actions_duration = target_total_duration_sim - BASE_DUREE_POST_FINAL
    min_core_actions_duration = sum_base_durations_for_mode * 0.7
    target_core_actions_duration = max(min_core_actions_duration, target_core_actions_duration)

    core_action_scaling_factor = target_core_actions_duration / sum_base_durations_for_mode

    def get_noisy_duration(base_duration, min_duration=0.05):
        scaled_duration = base_duration * core_action_scaling_factor
        noisy_duration = scaled_duration * (1 + np.random.uniform(-PHASE_VAR_FACTOR, PHASE_VAR_FACTOR))
        return max(min_duration, noisy_duration)

    duree_clic_bouton1_noisy = max(0.05, BASE_DUREE_CLIC_BOUTON_AVG * (1 + np.random.uniform(-RAND_FACTOR_CLIC, RAND_FACTOR_CLIC)))
    duree_redressement_charge = get_noisy_duration(BASE_DUREE_REDRESSEMENT_CHARGE, 0.5)
    duree_rotation = get_noisy_duration(BASE_DUREE_ROTATION, 0.5)
    duree_marche_avec_charge = get_noisy_duration(BASE_DUREE_MARCHE_AVEC_CHARGE, 2.0)
    duree_flexion_pose = get_noisy_duration(BASE_DUREE_FLEXION_POSE, 0.5)

    t_appui_bouton_blocage = 0.0
    t_fin_appui_blocage = t_appui_bouton_blocage + duree_clic_bouton1_noisy
    t_debut_redressement_charge = t_fin_appui_blocage
    t_fin_redressement_charge = t_debut_redressement_charge + duree_redressement_charge
    t_debut_deplacement_global = t_fin_redressement_charge
    t_fin_deplacement_global = t_debut_deplacement_global + duree_rotation + duree_marche_avec_charge
    t_debut_flexion_pose = t_fin_deplacement_global
    t_fin_flexion_pose = t_debut_flexion_pose + duree_flexion_pose
    t_charge_totalement_posee = t_fin_flexion_pose

    # CORRECTION: Définir angle_redressement_pitch_final ici pour qu'il soit dans la portée
    # en se basant sur la constante globale ANGLE_FLEXION_PITCH_POSE.
    # Cela ne change pas la logique, mais assure que la variable est définie.
    local_angle_redressement_pitch_final = -ANGLE_FLEXION_PITCH_POSE

    if generation_mode == "training":
        duree_avant_clic_deblocage = get_noisy_duration(BASE_DUREE_AVANT_CLIC_DEBLOCAGE, 0.1)
        duree_clic_bouton2_noisy = max(0.05, BASE_DUREE_CLIC_BOUTON_AVG * (1 + np.random.uniform(-RAND_FACTOR_CLIC, RAND_FACTOR_CLIC)))
        duree_redressement_sans_charge = get_noisy_duration(BASE_DUREE_REDRESSEMENT_SANS_CHARGE, 0.5)

        t_appui_bouton_deblocage = t_charge_totalement_posee + duree_avant_clic_deblocage
        t_fin_appui_deblocage = t_appui_bouton_deblocage + duree_clic_bouton2_noisy
        t_debut_redressement_sans_charge = t_fin_appui_deblocage
        t_fin_redressement_sans_charge = t_debut_redressement_sans_charge + duree_redressement_sans_charge
        duration_sim = t_fin_redressement_sans_charge + BASE_DUREE_POST_FINAL
    else:
        duree_clic_bouton2_noisy = 0
        t_appui_bouton_deblocage = t_charge_totalement_posee
        t_fin_appui_deblocage = t_appui_bouton_deblocage
        t_debut_redressement_sans_charge = t_fin_appui_deblocage
        t_fin_redressement_sans_charge = t_debut_redressement_sans_charge
        duration_sim = t_charge_totalement_posee + BASE_DUREE_POST_FINAL

    total_samples_sim = int(SAMPLING_RATE * duration_sim)
    time_vector_sim = np.linspace(0, duration_sim, total_samples_sim, endpoint=False)

    acc_x_c = np.zeros(total_samples_sim)
    acc_y_c = np.zeros(total_samples_sim)
    acc_z_c = np.full(total_samples_sim, G_ACCEL)
    gyro_x_c = np.zeros(total_samples_sim)
    gyro_y_c = np.zeros(total_samples_sim)
    gyro_z_c = np.zeros(total_samples_sim)
    force1_c = np.zeros(total_samples_sim)
    force2_c = np.zeros(total_samples_sim)
    btn_state_sim = np.zeros(total_samples_sim)

    btn_state_sim = generate_pulse(btn_state_sim, total_samples_sim, t_appui_bouton_blocage, t_fin_appui_blocage, 1)
    if generation_mode == "training":
        btn_state_sim = generate_pulse(btn_state_sim, total_samples_sim, t_appui_bouton_deblocage, t_fin_appui_deblocage, 1)

    duree_montee_force_sim = max(0.15, 0.3 * core_action_scaling_factor * (1 + np.random.uniform(-0.1, 0.1)))
    force1_c = generate_sigmoid_ramp(force1_c, total_samples_sim, t_fin_appui_blocage, t_fin_appui_blocage + duree_montee_force_sim, 0, FORCE_DUE_A_LA_CHARGE / 2)
    force2_c = generate_sigmoid_ramp(force2_c, total_samples_sim, t_fin_appui_blocage, t_fin_appui_blocage + duree_montee_force_sim, 0, FORCE_DUE_A_LA_CHARGE / 2)
    idx_maintien_s = int((t_fin_appui_blocage + duree_montee_force_sim) * SAMPLING_RATE)
    idx_maintien_e = int(t_debut_flexion_pose * SAMPLING_RATE)
    if idx_maintien_s < idx_maintien_e:
        force1_c = generate_pulse(force1_c, total_samples_sim, (t_fin_appui_blocage + duree_montee_force_sim), t_debut_flexion_pose, FORCE_DUE_A_LA_CHARGE / 2)
        force2_c = generate_pulse(force2_c, total_samples_sim, (t_fin_appui_blocage + duree_montee_force_sim), t_debut_flexion_pose, FORCE_DUE_A_LA_CHARGE / 2)
    force1_c = generate_sigmoid_ramp(force1_c, total_samples_sim, t_debut_flexion_pose, t_charge_totalement_posee, FORCE_DUE_A_LA_CHARGE / 2, 0)
    force2_c = generate_sigmoid_ramp(force2_c, total_samples_sim, t_debut_flexion_pose, t_charge_totalement_posee, FORCE_DUE_A_LA_CHARGE / 2, 0)
    idx_fn = int(t_charge_totalement_posee * SAMPLING_RATE)
    if idx_fn < total_samples_sim:
        force1_c[idx_fn:] = 0
        force2_c[idx_fn:] = 0

    acc_z_c = apply_acceleration_for_displacement(acc_z_c, total_samples_sim, t_debut_redressement_charge, t_fin_redressement_charge, DELTA_HAUTEUR_REDRESSEMENT1, is_z_axis=True)
    gyro_y_c = apply_angular_velocity_for_rotation(gyro_y_c, total_samples_sim, t_debut_redressement_charge, t_fin_redressement_charge, -INITIAL_ANGLE_PITCH)
    t_rot_s = t_debut_deplacement_global
    t_rot_e = t_rot_s + duree_rotation
    gyro_z_c = apply_angular_velocity_for_rotation(gyro_z_c, total_samples_sim, t_rot_s, t_rot_e, ANGLE_ROTATION_YAW)
    t_marche_s = t_rot_e
    t_marche_e = t_fin_deplacement_global
    acc_x_c = apply_acceleration_for_displacement(acc_x_c, total_samples_sim, t_marche_s, t_marche_e, DISTANCE_MARCHE)
    idx_m_s_y = int(t_marche_s * SAMPLING_RATE); idx_m_e_y = int(t_marche_e * SAMPLING_RATE)
    if idx_m_s_y < idx_m_e_y: acc_y_c[idx_m_s_y:idx_m_e_y] = 0

    idx_m_s_z = int(t_marche_s * SAMPLING_RATE); idx_m_e_z = int(t_marche_e * SAMPLING_RATE)
    if idx_m_s_z < idx_m_e_z and (t_marche_e - t_marche_s) > 0:
        num_pas = int((t_marche_e - t_marche_s) * (1.0 + np.random.uniform(-0.1, 0.1)))
        if num_pas > 0:
            for i in range(num_pas):
                ts_p = t_marche_s + i * (t_marche_e - t_marche_s) / num_pas
                te_p = ts_p + (t_marche_e - t_marche_s) / num_pas
                mid_p = ts_p + (te_p - ts_p) / 2
                amp_pas_z = 0.005 * (1 + np.random.uniform(-0.2, 0.2))
                acc_z_c = apply_acceleration_for_displacement(acc_z_c, total_samples_sim, ts_p, mid_p, amp_pas_z, is_z_axis=True)
                acc_z_c = apply_acceleration_for_displacement(acc_z_c, total_samples_sim, mid_p, te_p, -amp_pas_z, is_z_axis=True)

    idx_flex_s = int(t_debut_flexion_pose * SAMPLING_RATE); idx_flex_e = int(t_fin_flexion_pose * SAMPLING_RATE)
    if idx_flex_s < idx_flex_e:
        acc_x_c[idx_flex_s:idx_flex_e] = 0
        acc_y_c[idx_flex_s:idx_flex_e] = 0
    acc_z_c = apply_acceleration_for_displacement(acc_z_c, total_samples_sim, t_debut_flexion_pose, t_fin_flexion_pose, DELTA_HAUTEUR_FLEXION_POSE, is_z_axis=True)
    gyro_y_c = apply_angular_velocity_for_rotation(gyro_y_c, total_samples_sim, t_debut_flexion_pose, t_fin_flexion_pose, ANGLE_FLEXION_PITCH_POSE)

    if generation_mode == "training":
        idx_redress_s = int(t_debut_redressement_sans_charge * SAMPLING_RATE); idx_redress_e = int(t_fin_redressement_sans_charge * SAMPLING_RATE)
        if idx_redress_s < idx_redress_e:
            acc_x_c[idx_redress_s:idx_redress_e] = 0
            acc_y_c[idx_redress_s:idx_redress_e] = 0
        acc_z_c = apply_acceleration_for_displacement(acc_z_c, total_samples_sim, t_debut_redressement_sans_charge, t_fin_redressement_sans_charge, -DELTA_HAUTEUR_FLEXION_POSE, is_z_axis=True)
        gyro_y_c = apply_angular_velocity_for_rotation(gyro_y_c, total_samples_sim, t_debut_redressement_sans_charge, t_fin_redressement_sans_charge, local_angle_redressement_pitch_final) # Utilisation de la variable locale

    idx_c_s = int(t_fin_redressement_sans_charge * SAMPLING_RATE if generation_mode == "training" else t_charge_totalement_posee * SAMPLING_RATE)
    if idx_c_s < total_samples_sim:
        acc_x_c[idx_c_s:] = 0; acc_y_c[idx_c_s:] = 0; acc_z_c[idx_c_s:] = G_ACCEL
        gyro_x_c[idx_c_s:] = 0; gyro_y_c[idx_c_s:] = 0; gyro_z_c[idx_c_s:] = 0

    acc_x_n = acc_x_c + np.random.normal(0, NOISE_STD_ACC_BASE, total_samples_sim)
    acc_y_n = acc_y_c + np.random.normal(0, NOISE_STD_ACC_BASE, total_samples_sim)
    acc_z_n = acc_z_c + np.random.normal(0, NOISE_STD_ACC_BASE, total_samples_sim)
    gyro_x_n = gyro_x_c + np.random.normal(0, NOISE_STD_GYRO_BASE, total_samples_sim)
    gyro_y_n = gyro_y_c + np.random.normal(0, NOISE_STD_GYRO_BASE, total_samples_sim)
    gyro_z_n = gyro_z_c + np.random.normal(0, NOISE_STD_GYRO_BASE, total_samples_sim)
    force1_n = np.maximum(0, force1_c + np.random.normal(0, NOISE_STD_FORCE_BASE, total_samples_sim))
    force2_n = np.maximum(0, force2_c + np.random.normal(0, NOISE_STD_FORCE_BASE, total_samples_sim))

    acc_motion_x_n = acc_x_n.copy(); acc_motion_y_n = acc_y_n.copy(); acc_motion_z_n = acc_z_n - G_ACCEL
    vel_x_n = cumulative_trapezoid(acc_motion_x_n, dx=DT, initial=0)
    vel_y_n = cumulative_trapezoid(acc_motion_y_n, dx=DT, initial=0)
    vel_z_n = cumulative_trapezoid(acc_motion_z_n, dx=DT, initial=0)
    pos_x_n = INITIAL_POS_X + cumulative_trapezoid(vel_x_n, dx=DT, initial=0)
    pos_y_n = INITIAL_POS_Y + cumulative_trapezoid(vel_y_n, dx=DT, initial=0)
    pos_z_n = INITIAL_POS_Z + cumulative_trapezoid(vel_z_n, dx=DT, initial=0)
    angle_roll_n = INITIAL_ANGLE_ROLL + cumulative_trapezoid(gyro_x_n, dx=DT, initial=0)
    angle_pitch_n = INITIAL_ANGLE_PITCH + cumulative_trapezoid(gyro_y_n, dx=DT, initial=0)
    angle_yaw_n = INITIAL_ANGLE_YAW + cumulative_trapezoid(gyro_z_n, dx=DT, initial=0)

    df = pd.DataFrame({
        'Time (s)': time_vector_sim,
        'Acc_X_Raw (m/s^2)': acc_x_n, 'Acc_Y_Raw (m/s^2)': acc_y_n, 'Acc_Z_Raw (m/s^2)': acc_z_n,
        'Gyro_X (rad/s)': gyro_x_n, 'Gyro_Y (rad/s)': gyro_y_n, 'Gyro_Z (rad/s)': gyro_z_n,
        'Force_Cable1 (N)': force1_n, 'Force_Cable2 (N)': force2_n, 'Button_State': btn_state_sim,
        'Vel_X_Noisy (m/s)': vel_x_n, 'Vel_Y_Noisy (m/s)': vel_y_n, 'Vel_Z_Noisy (m/s)': vel_z_n,
        'Pos_X_Noisy (m)': pos_x_n, 'Pos_Y_Noisy (m)': pos_y_n, 'Pos_Z_Noisy (m)': pos_z_n,
        'Angle_Roll_Noisy (rad)': angle_roll_n, 'Angle_Pitch_Noisy (rad)': angle_pitch_n, 'Angle_Yaw_Noisy (rad)': angle_yaw_n,
    })
    print(f"Simulation {simulation_id} générée. Durée cible: {target_total_duration_sim:.2f}s, Durée réelle: {duration_sim:.2f}s. Clic1: {duree_clic_bouton1_noisy:.3f}s, Clic2: {duree_clic_bouton2_noisy:.3f}s (si training)")
    return df

# --- Bloc 3: Fonction de Traçage pour une Simulation ---
def plot_simulation_data(simulation_id, data_df=None, output_dir="simulation_outputs", gen_mode_for_plot="training"):
    """Charge et trace les données d'une simulation spécifique."""
    if data_df is None:
        filepath = os.path.join(output_dir, f"simulation_{simulation_id}_{gen_mode_for_plot}.xlsx")
        if not os.path.exists(filepath):
            print(f"Erreur: Le fichier {filepath} n'existe pas.")
            filepath_old = os.path.join(output_dir, f"simulation_{simulation_id}.xlsx") # Fallback
            if not os.path.exists(filepath_old):
                print(f"Erreur: Le fichier {filepath_old} n'existe pas non plus.")
                return
            else:
                filepath = filepath_old
                print(f"Utilisation du fichier avec ancien nommage: {filepath}")
        try:
            data_df = pd.read_excel(filepath)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {filepath}: {e}")
            return

    fig, axs = plt.subplots(6, 1, figsize=(15, 22), sharex=True)
    fig.suptitle(f'Données Synthétiques - Simulation {simulation_id} (Mode: {gen_mode_for_plot})', fontsize=16)

    axs[0].plot(data_df['Time (s)'], data_df['Vel_X_Noisy (m/s)'], label='Vitesse Linéaire X (m/s)')
    axs[0].plot(data_df['Time (s)'], data_df['Vel_Y_Noisy (m/s)'], label='Vitesse Linéaire Y (m/s)')
    axs[0].plot(data_df['Time (s)'], data_df['Vel_Z_Noisy (m/s)'], label='Vitesse Linéaire Z (m/s)')
    axs[0].set_ylabel('Vitesse Linéaire (m/s)'); axs[0].legend(loc='upper right'); axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].set_title('Vitesses Linéaires (Estimées à partir de données bruitées)')

    axs[1].plot(data_df['Time (s)'], data_df['Gyro_X (rad/s)'], label='Vitesse Angulaire X (rad/s)')
    axs[1].plot(data_df['Time (s)'], data_df['Gyro_Y (rad/s)'], label='Vitesse Angulaire Y (rad/s)')
    axs[1].plot(data_df['Time (s)'], data_df['Gyro_Z (rad/s)'], label='Vitesse Angulaire Z (rad/s)')
    axs[1].set_ylabel('Vitesse Angulaire (rad/s)'); axs[1].legend(loc='upper right'); axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].set_title('Vitesses Angulaires (Gyromètre Brutes)')

    axs[2].plot(data_df['Time (s)'], data_df['Force_Cable1 (N)'], label='Force Câble 1 (N)')
    axs[2].plot(data_df['Time (s)'], data_df['Force_Cable2 (N)'], label='Force Câble 2 (N)', linestyle='--')
    axs[2].set_ylabel('Force (N)'); axs[2].legend(loc='upper right'); axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].set_title('Capteurs de Force dans les Câbles (Bruités)')

    axs[3].plot(data_df['Time (s)'], data_df['Button_State'], label='État Bouton', drawstyle='steps-post', color='red')
    axs[3].set_ylabel('État Bouton (0 ou 1)'); axs[3].legend(loc='upper right'); axs[3].grid(True, linestyle='--', alpha=0.7)
    axs[3].set_title('État du Bouton de Commande'); axs[3].set_yticks([0, 1]); axs[3].set_yticklabels(['Non Appuyé (0)', 'Appuyé (1)'])

    axs[4].plot(data_df['Time (s)'], data_df['Pos_X_Noisy (m)'], label='Position X (m)')
    axs[4].plot(data_df['Time (s)'], data_df['Pos_Y_Noisy (m)'], label='Position Y (m)')
    axs[4].plot(data_df['Time (s)'], data_df['Pos_Z_Noisy (m)'], label='Position Z (m)')
    axs[4].set_ylabel('Position Linéaire (m)'); axs[4].legend(loc='upper right'); axs[4].grid(True, linestyle='--', alpha=0.7)
    axs[4].set_title('Positions Linéaires (Estimées à partir de données bruitées)')

    axs[5].plot(data_df['Time (s)'], np.rad2deg(data_df['Angle_Roll_Noisy (rad)']), label='Angle Roll (°)')
    axs[5].plot(data_df['Time (s)'], np.rad2deg(data_df['Angle_Pitch_Noisy (rad)']), label='Angle Pitch (°)')
    axs[5].plot(data_df['Time (s)'], np.rad2deg(data_df['Angle_Yaw_Noisy (rad)']), label='Angle Yaw (°)')
    axs[5].set_ylabel('Position Angulaire (°)')
    axs[5].legend(loc='upper right'); axs[5].grid(True, linestyle='--', alpha=0.7)
    axs[5].set_title('Positions Angulaires (Estimées à partir de données bruitées)')

    button_press_times = data_df['Time (s)'][data_df['Button_State'].diff().fillna(0) > 0].tolist()
    common_vline_params = {'color': 'dimgray', 'linestyle': ':', 'lw': 1.5, 'alpha':0.8}
    if len(button_press_times) > 0:
        t_blocage_plot = button_press_times[0]
        axs[0].axvline(t_blocage_plot, **common_vline_params, label=f'Blocage ({t_blocage_plot:.2f}s)')
        for i_ax in range(1, 6): axs[i_ax].axvline(t_blocage_plot, **common_vline_params)
    if gen_mode_for_plot == "training" and len(button_press_times) > 1:
        t_deblocage_plot = button_press_times[1]
        axs[0].axvline(t_deblocage_plot, color='darkorange', linestyle=':', lw=1.5, alpha=0.8, label=f'Déblocage ({t_deblocage_plot:.2f}s)')
        for i_ax in range(1, 6): axs[i_ax].axvline(t_deblocage_plot, color='darkorange', linestyle=':', lw=1.5, alpha=0.8)

    handles, labels = axs[0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.005))
    for i_ax in range(6): axs[i_ax].set_xlabel('Temps (s)' if i_ax == 5 else '')
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.show()

# --- Bloc 4 & 5: Script Principal (Génération et Traçage Interactif) ---
if __name__ == "__main__":
    num_simulations_a_generer = 0
    while True:
        try:
            num_str = input("Combien de simulations souhaitez-vous générer ? ")
            num_simulations_a_generer = int(num_str)
            if num_simulations_a_generer > 0: break
            else: print("Veuillez entrer un nombre positif.")
        except ValueError: print("Entrée invalide. Veuillez entrer un nombre entier.")

    gen_mode_input = ""
    while gen_mode_input not in ["training", "inference"]:
        gen_mode_input = input("Mode de génération ('training' ou 'inference') ? ").lower()
        if gen_mode_input not in ["training", "inference"]:
            print("Mode invalide. Veuillez choisir 'training' ou 'inference'.")

    output_directory = "simulation_outputs" # Vous pouvez changer cela pour "training_dataset" ou "inference_dataset" au besoin
    if gen_mode_input == "training":
        output_directory = "training_dataset"
    elif gen_mode_input == "inference":
        output_directory = "inference_dataset"

    os.makedirs(output_directory, exist_ok=True)
    print(f"Les fichiers Excel seront sauvegardés dans : {os.path.abspath(output_directory)}")

    all_simulation_data_dfs = []

    for i in range(1, num_simulations_a_generer + 1):
        sim_data_df = generate_single_simulation_data(i, generation_mode=gen_mode_input)
        all_simulation_data_dfs.append(sim_data_df)

        excel_filename = f"simulation_{i}_{gen_mode_input}.xlsx"
        excel_filepath = os.path.join(output_directory, excel_filename)
        try:
            sim_data_df.to_excel(excel_filepath, index=False, engine='openpyxl')
            print(f"Simulation {i} ({gen_mode_input}) sauvegardée dans {excel_filepath}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la simulation {i} en Excel: {e}")
        print("-" * 30)

    print(f"\n{num_simulations_a_generer} simulations ({gen_mode_input}) générées et sauvegardées.")

    last_gen_mode_for_plotting = gen_mode_input
    last_output_dir_for_plotting = output_directory

    while True:
        plot_choice_str = input(f"Souhaitez-vous tracer une simulation spécifique (mode '{last_gen_mode_for_plotting}') ? (numéro ou 'non') ")
        if plot_choice_str.lower() == 'non': break
        try:
            sim_to_plot = int(plot_choice_str)
            if 1 <= sim_to_plot <= num_simulations_a_generer:
                # Tenter de plotter à partir des données en mémoire si elles existent et correspondent
                if sim_to_plot -1 < len(all_simulation_data_dfs) and gen_mode_input == last_gen_mode_for_plotting :
                     plot_simulation_data(sim_to_plot, data_df=all_simulation_data_dfs[sim_to_plot-1], output_dir=last_output_dir_for_plotting, gen_mode_for_plot=last_gen_mode_for_plotting)
                else:
                    print(f"Données pour la simulation {sim_to_plot} (mode {last_gen_mode_for_plotting}) non trouvées en mémoire ou mode différent. Tentative de chargement depuis le fichier.")
                    plot_simulation_data(sim_to_plot, data_df=None, output_dir=last_output_dir_for_plotting, gen_mode_for_plot=last_gen_mode_for_plotting)
            else:
                print(f"Numéro de simulation invalide. Entrez un nombre entre 1 et {num_simulations_a_generer}.")
        except ValueError: print("Entrée invalide.")
        except Exception as e_plot: print(f"Erreur lors du traçage: {e_plot}")