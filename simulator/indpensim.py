import numpy as np
from scipy.integrate import solve_ivp
import warnings

# --- FONCTIONS STUBS / UTILITAIRES ---

def create_batch(h, T):
    """
    Crée la structure de données vide pour stocker les résultats de simulation.
    Remplace le fichier createBatch.m manquant.
    """
    num_steps = int(T / h) + 1
    t_vec = np.linspace(0, T, num_steps)
    
    # Liste des variables d'état et de contrôle
    vars_list = [
        'S', 'DO2', 'O2', 'P', 'V', 'Wt', 'pH', 'T', 'Q', 'Viscosity', 
        'Culture_age', 'a0', 'a1', 'a3', 'a4', 
        'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'nm', 'phi0',
        'CO2outgas', 'CO2_d', 'PAA', 'NH3', 'mu_P_calc', 'mu_X_calc', 'X',
        'Fg', 'RPM', 'Fpaa', 'Fs', 'Fa', 'Fb', 'Fc', 'Foil', 'Fh', 'Fw', 'pressure', 'Fremoved',
        'Fault_ref', 'Control_ref', 'PAT_ref', 'Batch_ref',
        'OUR', 'CER',
        'NH3_offline', 'Viscosity_offline', 'PAA_offline', 'P_offline', 'X_offline'
    ]
    
    batch_struct = {}
    for var in vars_list:
        batch_struct[var] = {
            'y': np.zeros(num_steps),
            't': np.zeros(num_steps)
        }
    return batch_struct

def raman_sim(k, X, h, T):
    """Stub pour Raman_Sim.m"""
    return X

def substrate_prediction(k, X, h, T):
    """Stub pour Substrate_prediction.m"""
    return X

# Import de la fonction ODE (relative import for package)
from .indpensim_ode import indpensim_ode

# --- FONCTION PRINCIPALE ---

def indpensim(f_input, Xd, x0, h, T, solv, p, Ctrl_flags):
    """
    Fonction principale de simulation IndPenSim.
    
    Args:
        f_input: Fonction de contrôle (callback)
        Xd: Données externes / Perturbations
        x0: Conditions initiales (Dict)
        h: Pas de temps d'échantillonnage (heures)
        T: Durée totale (heures)
        solv: Type de solveur (1=RK45, 2=BDF/Stiff, 3=LSODA)
        p: Paramètres du modèle
        Ctrl_flags: Flags de simulation
    """
    
    # Simulation timing init
    N = int(T / h) # experiment length in samples
    h_ode = h / 20 # ode solver step size (hours)
    t_vec = np.arange(0, T + h/1000, h) # + epsilon pour inclure la fin
    
    # Creates batch structure
    X = create_batch(h, T)
    
    # Converts from pH to H+ conc.
    current_pH_conc = 10**(-x0['pH']) 
    
    print(f"Démarrage simulation IndPenSim (N={N} pas)...")
    
    for k in range(len(t_vec)):
        
        # Step 1: Initial Conditions (k=0 en Python equiv k=1 en MATLAB)
        if k == 0:
            X['S']['y'][0] = x0['S']
            X['DO2']['y'][0] = x0['DO2']
            X['X']['y'][0] = x0['X']
            X['P']['y'][0] = x0['P']
            X['V']['y'][0] = x0['V']
            X['CO2outgas']['y'][0] = x0['CO2outgas']
            X['pH']['y'][0] = current_pH_conc
            X['T']['y'][0] = x0['T']
            
            # Initialisation des temps
            for key in X:
                X[key]['t'][0] = 0

        # Step 2: Get Control Inputs (MVs)
        u, X = f_input(X, Xd, k, h, T, Ctrl_flags)

        # Step 3: Build Initial Conditions for ODE (x00)
        if k == 0:
            x00 = np.array([
                x0['S'], x0['DO2'], x0['O2'], x0['P'], x0['V'], x0['Wt'], 
                current_pH_conc, x0['T'], 0, 4, x0['Culture_age'], 
                x0['a0'], x0['a1'], x0['a3'], x0['a4']
            ] + [0]*10 + [
                0, 0, x0['CO2outgas'], 0, x0['PAA'], x0['NH3'], 0, 0
            ])
            pass
            
        else:
            prev = k - 1
            x00 = np.array([
                X['S']['y'][prev], X['DO2']['y'][prev], X['O2']['y'][prev], X['P']['y'][prev],
                X['V']['y'][prev], X['Wt']['y'][prev], X['pH']['y'][prev], X['T']['y'][prev],
                X['Q']['y'][prev], X['Viscosity']['y'][prev], X['Culture_age']['y'][prev],
                X['a0']['y'][prev], X['a1']['y'][prev], X['a3']['y'][prev], X['a4']['y'][prev],
                X['n0']['y'][prev], X['n1']['y'][prev], X['n2']['y'][prev], X['n3']['y'][prev],
                X['n4']['y'][prev], X['n5']['y'][prev], X['n6']['y'][prev], X['n7']['y'][prev],
                X['n8']['y'][prev], X['n9']['y'][prev], X['nm']['y'][prev], X['phi0']['y'][prev],
                X['CO2outgas']['y'][prev], X['CO2_d']['y'][prev], X['PAA']['y'][prev], X['NH3']['y'][prev],
                0, 0 
            ])
            
        # Step 4: Process Disturbances
        distMuP = Xd['distMuP']['y'][k]
        distMuX = Xd['distMuX']['y'][k]
        distcs  = Xd['distcs']['y'][k]
        distcoil = Xd['distcoil']['y'][k]
        distabc = Xd['distabc']['y'][k]
        distPAA = Xd['distPAA']['y'][k]
        distTcin = Xd['distTcin']['y'][k]
        distO_2in = Xd['distO_2in']['y'][k]

        # Input Vector u00
        u00 = np.array([
            Ctrl_flags['Inhib'], u['Fs'], u['Fg'], u['RPM'], u['Fc'], u['Fh'], u['Fb'], u['Fa'],
            h_ode, u['Fw'], u['pressure'], u['viscosity'], u['Fremoved'], u['Fpaa'], u['Foil'],
            u.get('NH3_shots', 0),
            Ctrl_flags['Dis'], distMuP, distMuX, distcs, distcoil, distabc, distPAA, distTcin, distO_2in,
            Ctrl_flags['Vis']
        ])

        # Step 5: Special Logic for Inhibition / Stability
        if Ctrl_flags['Inhib'] in [1, 2] and k > 65:
            mu_x_slice = X['mu_X_calc']['y'][k-65 : k]
            a1 = np.diff(mu_x_slice)
            a2 = a1 < 0
            if np.sum(a2) >= 63:
                if isinstance(p, list) or isinstance(p, np.ndarray):
                    p_current = p.copy() if isinstance(p, np.ndarray) else list(p)
                    p_current[1] = X['mu_X_calc']['y'][k-1] * 5
                    p = p_current

        # Step 6: Solve ODE
        t_span = (t_vec[k], t_vec[k] + h)
        
        method = 'RK45'
        if solv == 2:
            method = 'BDF'
        elif solv == 3:
            method = 'Radau'
            
        sol = solve_ivp(indpensim_ode, t_span, x00, args=(u00, p), method=method)
        
        y_final = sol.y[:, -1]
        y_final[y_final <= 0] = 0.001
        t_final = sol.t[-1]

        # Step 7: Saving Results
        
        # MVs
        X['Fg']['y'][k] = u['Fg']; X['Fg']['t'][k] = t_final
        X['RPM']['y'][k] = u['RPM']; X['RPM']['t'][k] = t_final
        X['Fpaa']['y'][k] = u['Fpaa']; X['Fpaa']['t'][k] = t_final
        X['Fs']['y'][k] = u['Fs']; X['Fs']['t'][k] = t_final
        X['Fa']['y'][k] = u['Fa']; X['Fa']['t'][k] = t_final
        X['Fb']['y'][k] = u['Fb']; X['Fb']['t'][k] = t_final
        X['Fc']['y'][k] = u['Fc']; X['Fc']['t'][k] = t_final
        X['Foil']['y'][k] = u['Foil']; X['Foil']['t'][k] = t_final
        X['Fh']['y'][k] = u['Fh']; X['Fh']['t'][k] = t_final
        X['Fw']['y'][k] = u['Fw']; X['Fw']['t'][k] = t_final
        X['pressure']['y'][k] = u['pressure']; X['pressure']['t'][k] = t_final
        X['Fremoved']['y'][k] = u['Fremoved']; X['Fremoved']['t'][k] = t_final

        # States
        X['S']['y'][k] = y_final[0]; X['S']['t'][k] = t_final
        
        do2_val = y_final[1]
        X['DO2']['y'][k] = 1 if do2_val < 2 else do2_val
        X['DO2']['t'][k] = t_final
        
        X['O2']['y'][k] = y_final[2]; X['O2']['t'][k] = t_final
        X['P']['y'][k] = y_final[3]; X['P']['t'][k] = t_final
        X['V']['y'][k] = y_final[4]; X['V']['t'][k] = t_final
        X['Wt']['y'][k] = y_final[5]; X['Wt']['t'][k] = t_final
        X['pH']['y'][k] = y_final[6]; X['pH']['t'][k] = t_final
        X['T']['y'][k] = y_final[7]; X['T']['t'][k] = t_final
        X['Q']['y'][k] = y_final[8]; X['Q']['t'][k] = t_final
        X['Viscosity']['y'][k] = y_final[9]; X['Viscosity']['t'][k] = t_final
        X['Culture_age']['y'][k] = y_final[10]; X['Culture_age']['t'][k] = t_final
        
        X['a0']['y'][k] = y_final[11]; X['a0']['t'][k] = t_final
        X['a1']['y'][k] = y_final[12]; X['a1']['t'][k] = t_final
        X['a3']['y'][k] = y_final[13]; X['a3']['t'][k] = t_final
        X['a4']['y'][k] = y_final[14]; X['a4']['t'][k] = t_final
        
        for i_n in range(10):
            X[f'n{i_n}']['y'][k] = y_final[15 + i_n]
            X[f'n{i_n}']['t'][k] = t_final
            
        X['nm']['y'][k] = y_final[25]; X['nm']['t'][k] = t_final
        X['phi0']['y'][k] = y_final[26]; X['phi0']['t'][k] = t_final
        X['CO2outgas']['y'][k] = y_final[27]; X['CO2outgas']['t'][k] = t_final
        X['CO2_d']['y'][k] = y_final[28]; X['CO2_d']['t'][k] = t_final
        X['PAA']['y'][k] = y_final[29]; X['PAA']['t'][k] = t_final
        X['NH3']['y'][k] = y_final[30]; X['NH3']['t'][k] = t_final
        
        X['mu_P_calc']['y'][k] = y_final[31]; X['mu_P_calc']['t'][k] = t_final
        X['mu_X_calc']['y'][k] = y_final[32]; X['mu_X_calc']['t'][k] = t_final
        
        # Calculated States
        X['X']['y'][k] = X['a0']['y'][k] + X['a1']['y'][k] + X['a3']['y'][k] + X['a4']['y'][k]
        X['X']['t'][k] = t_final
        
        X['Fault_ref']['y'][k] = u.get('Fault_ref', 0)
        X['Fault_ref']['t'][k] = t_final
        X['Control_ref']['y'][k] = Ctrl_flags['PRBS']
        X['Control_ref']['t'][k] = Ctrl_flags['Batch_Num']
        X['PAT_ref']['y'][k] = Ctrl_flags['Raman_spec']
        X['PAT_ref']['t'][k] = Ctrl_flags['Batch_Num']
        X['Batch_ref']['y'][k] = Ctrl_flags['Batch_Num']
        X['Batch_ref']['t'][k] = Ctrl_flags['Batch_Num']

        # Step 8: OUR / CER Calculation
        O2_in = 0.204
        
        denom_our = (1 - X['O2']['y'][k] - X['CO2outgas']['y'][k]/100)
        if denom_our == 0: denom_our = 1e-6
        
        term_our = (0.7902 / denom_our)
        X['OUR']['y'][k] = (32 * X['Fg']['y'][k] / 22.4) * (O2_in - X['O2']['y'][k] * term_our)
        X['OUR']['t'][k] = t_final
        
        denom_cer = (1 - O2_in - X['CO2outgas']['y'][k]/100)
        if denom_cer == 0: denom_cer = 1e-6
        
        term_cer = (0.7902 / denom_cer - 0.0330)
        X['CER']['y'][k] = (44 * X['Fg']['y'][k] / 22.4) * ((0.65 * X['CO2outgas']['y'][k] / 100) * term_cer)
        X['CER']['t'][k] = t_final

        # Step 9: Advanced sensors
        if k > 10:
            if Ctrl_flags['Raman_spec'] == 1:
                X = raman_sim(k, X, h, T)
            elif Ctrl_flags['Raman_spec'] == 2:
                X = raman_sim(k, X, h, T)
                X = substrate_prediction(k, X, h, T)

        # Step 10: Off-line measurements logic
        is_sampling_time = (np.isclose(t_final % Ctrl_flags['Off_line_m'], 0, atol=1e-3) or 
                            np.isclose(t_final, T, atol=1e-3) or 
                            np.isclose(t_final, 1, atol=1e-3))
        
        if is_sampling_time:
            delay_steps = int(Ctrl_flags['Off_line_delay'] / h)
            idx_delayed = k - delay_steps
            
            if idx_delayed >= 0:
                X['NH3_offline']['y'][k] = X['NH3']['y'][idx_delayed]
                X['NH3_offline']['t'][k] = X['NH3']['t'][idx_delayed]
                X['Viscosity_offline']['y'][k] = X['Viscosity']['y'][idx_delayed]
                X['Viscosity_offline']['t'][k] = X['Viscosity']['t'][idx_delayed]
                X['PAA_offline']['y'][k] = X['PAA']['y'][idx_delayed]
                X['PAA_offline']['t'][k] = X['PAA']['t'][idx_delayed]
                X['P_offline']['y'][k] = X['P']['y'][idx_delayed]
                X['P_offline']['t'][k] = X['P']['t'][idx_delayed]
                X['X_offline']['y'][k] = X['X']['y'][idx_delayed]
                X['X_offline']['t'][k] = X['X']['t'][idx_delayed]
            else:
                for var in ['NH3_offline', 'Viscosity_offline', 'PAA_offline', 'P_offline', 'X_offline']:
                    X[var]['y'][k] = np.nan
                    X[var]['t'][k] = np.nan
        else:
            for var in ['NH3_offline', 'Viscosity_offline', 'PAA_offline', 'P_offline', 'X_offline']:
                X[var]['y'][k] = np.nan
                X[var]['t'][k] = np.nan

    # Fin de boucle
    
    # Unit conversions post-sim
    X['pH']['y'] = -np.log10(X['pH']['y'] + 1e-9) 
    X['Q']['y'] = X['Q']['y'] / 1000

    print("Simulation terminée.")
    return X
