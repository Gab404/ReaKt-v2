import numpy as np
from scipy.signal import lfilter
from .parameter_list import parameter_list
from .fctrl_indpensim import fctrl_indpensim
from .indpensim import indpensim

# --- FONCTIONS UTILITAIRES ---
def create_channel(name, y_unit, t_unit, t, y):
    """Fonction utilitaire pour structurer les données comme dans MATLAB"""
    return {
        'name': name,
        'yUnit': y_unit,
        'tUnit': t_unit,
        't': t,
        'y': y
    }

# --- DÉBUT DE LA FONCTION PRINCIPALE ---

def indpensim_run(Batch_no, Batch_run_flags):
    """
    Convertit indpensim_run.m
    Initialise et lance une simulation de fermentation.
    """
    
    # Initialisation des Flags
    Ctrl_flags = {}
    Ctrl_flags['SBC'] = Batch_run_flags['Control_strategy'][Batch_no - 1]
    Ctrl_flags['PRBS'] = Batch_run_flags['Control_strategy'][Batch_no - 1]
    Ctrl_flags['Fixed_Batch_length'] = Batch_run_flags['Batch_length'][Batch_no - 1]
    
    Ctrl_flags['IC'] = 0      # 0 - Random initial conditions
    Ctrl_flags['Inhib'] = 2   # 2 - Full inhibition model
    Ctrl_flags['Dis'] = 1     # 1 - Process disturbances enabled
    Ctrl_flags['Faults'] = Batch_run_flags['Batch_fault_order_reference'][Batch_no - 1]
    Ctrl_flags['Vis'] = 0     # 0 - Simulated viscosity
    Ctrl_flags['Raman_spec'] = Batch_run_flags['Raman_spec'][Batch_no - 1]
    
    Ctrl_flags['Batch_Num'] = Batch_no
    Ctrl_flags['Off_line_m'] = 12    # hours
    Ctrl_flags['Off_line_delay'] = 4 # hours
    Ctrl_flags['plots'] = 1

    # Dictionnaire pour les conditions initiales
    x0 = {} 
    
    # Standard batch simulation logic
    if Ctrl_flags['IC'] == 0:
        Ctrl_flags['SBC'] = 0
        Ctrl_flags['Vis'] = 0
        Optimum_Batch_lenght = 230 # hours
        
        if Ctrl_flags['Fixed_Batch_length'] == 1:
            Batch_length_variation = 25 * np.random.randn()
            T = Optimum_Batch_lenght + Batch_length_variation
            T = round(T)
        else:
            T = Optimum_Batch_lenght
            
        Randomise_each_bactch = 1
        if Randomise_each_bactch == 1:
            Random_seed_ref = int(np.ceil(np.random.rand() * 1000))
        else:
            Random_seed_ref = 5
            
        Seed_ref = 31 + Random_seed_ref
        Rand_ref = 1
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        intial_conds = 0.5 + 0.05 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['mux'] = 0.41 + 0.025 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['mup'] = 0.041 + 0.0025 * np.random.randn()
        
        h = 0.2 # Sampling rate (12 mins = 0.2h)
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['S'] = 1 + 0.1 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['DO2'] = 15 + 0.5 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['X'] = intial_conds + 0.1 * np.random.randn()
        
        x0['P'] = 0
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['V'] = 5.800e+04 + 500 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['Wt'] = 6.2e+04 + 500 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['CO2outgas'] = 0.038 + 0.001 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['O2'] = 0.20 + 0.05 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['pH'] = 6.5 + 0.1 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['T'] = 297 + 0.5 * np.random.randn()
        
        x0['a0'] = intial_conds * (1/3)
        x0['a1'] = intial_conds * (2/3)
        x0['a3'] = 0
        x0['a4'] = 0
        x0['Culture_age'] = 0
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['PAA'] = 1400 + 50 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        x0['NH3'] = 1700 + 50 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        alpha_kla = 85 + 10 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref); Rand_ref += 1
        PAA_c = 530000 + 20000 * np.random.randn()
        
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        N_conc_paa = 2 * 75000 + 2000 * np.random.randn()
        
        Batch_time = np.arange(0, T + h, h)
        if Batch_time[-1] > T:
             Batch_time = Batch_time[:-1]

        # Set points
        Ctrl_flags['T_sp'] = 298
        Ctrl_flags['pH_sp'] = 6.5

    # Reset RNG for disturbances
    np.random.seed(Random_seed_ref + Batch_no)
    
    # --- PROCESS DISTURBANCES ---
    b1 = np.array([1 - 0.995])
    a1 = np.array([1, -0.995])
    
    num_samples = len(Batch_time)
    Xinterp = {}
    
    def generate_disturbance(scale_factor, name, unit):
        v = np.random.randn(num_samples)
        dist = lfilter(b1, a1, scale_factor * v)
        return create_channel(name, unit, 'h', Batch_time, dist)

    Xinterp['distMuP']   = generate_disturbance(0.03,      'Penicillin specific growth rate disturbance', 'g/Lh')
    Xinterp['distMuX']   = generate_disturbance(0.25,      'Biomass specific growth rate disturbance',    'hr^{-1}')
    Xinterp['distcs']    = generate_disturbance(5 * 300,   'Substrate concentration disturbance',         'g L^{-1}')
    Xinterp['distcoil']  = generate_disturbance(300,       'Substrate concentration disturbance',         'g L^{-1}')
    Xinterp['distabc']   = generate_disturbance(0.2,       'Acid/Base concentration disturbance',         'g L^{-1}')
    Xinterp['distPAA']   = generate_disturbance(300000,    'Phenylacetic acid concentration disturbance', 'g L^{-1}')
    Xinterp['distTcin']  = generate_disturbance(100,       'Coolant inlet temperature disturbance',       'K')
    Xinterp['distO_2in'] = generate_disturbance(0.02,      'Oxygen inlet concentration',                  '%')

    # --- EXECUTION DE LA SIMULATION ---
    par = parameter_list(x0, alpha_kla, N_conc_paa, PAA_c)
    
    print('Running IndPenSim (Python)...')
    
    Xref = indpensim(fctrl_indpensim, Xinterp, x0, h, T, 2, par, Ctrl_flags)
    
    if Ctrl_flags['Raman_spec'] > 1:
        pass

    # --- CALCUL DES STATISTIQUES ---
    if 'Stats' not in Xref:
        Xref['Stats'] = {}
        
    penicillin_harvested_during = np.sum(Xref['Fremoved']['y'] * Xref['P']['y']) * h
    penicillin_harvested_end = Xref['V']['y'][-1] * Xref['P']['y'][-1]
    
    Xref['Stats']['Penicllin_harvested_during_batch'] = penicillin_harvested_during
    Xref['Stats']['Penicllin_harvested_end_of_batch'] = penicillin_harvested_end
    Xref['Stats']['Penicllin_yield_total'] = penicillin_harvested_end - penicillin_harvested_during
    Xref['Stats']['Batch_length'] = Xref['V']['t'][-1] if 't' in Xref['V'] else T

    print(f"Penicillin harvested during the batch {round(Xref['Stats']['Penicllin_harvested_during_batch']/1000)} Kg")
    print(f"Final Penicillin yield at harvest {round(Xref['Stats']['Penicllin_harvested_end_of_batch']/1000)} Kg")
    print(f"Total penicillin {round(Xref['Stats']['Penicllin_yield_total']/1000)} Kg")

    return Xref
