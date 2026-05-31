import numpy as np

def indpensim_ode(t, y, inp1, par):
    """
    Ordinary Differential Equations for IndPenSim.
    
    Args:
        t: Time (scalar)
        y: State vector (numpy array)
        inp1: Input parameters vector (Control variables & Flags)
        par: Model parameters vector
        
    Returns:
        dy: Derivative vector (numpy array)
    """
    
    # --- 1. Parameter Unpacking (par) ---
    # Indices shifted by -1 compared to MATLAB
    mu_p = par[0]
    mux_max = par[1]
    ratio_mu_e_mu_b = par[2]
    P_std_dev = par[3]
    mean_P = par[4]
    mu_v = par[5]
    mu_a = par[6]
    mu_diff = par[7]
    beta_1 = par[8]
    K_b = par[9]
    K_diff = par[10]
    K_diff_L = par[11]
    K_e = par[12]
    K_v = par[13]
    delta_r = par[14]
    k_v = par[15]
    D = par[16]
    rho_a0 = par[17]
    rho_d = par[18]
    mu_h = par[19]
    r_0 = par[20]
    delta_0 = par[21]

    # Process related parameters
    Y_sX = par[22]
    Y_sP = par[23]
    m_s = par[24]
    c_oil = par[25]
    c_s = par[26]
    Y_O2_X = par[27]
    Y_O2_P = par[28]
    m_O2_X = par[29]
    alpha_kla_par = par[30] # Renamed to avoid confusion if local var exists
    a_par = par[31]
    b_par = par[32]
    c_par = par[33]
    d_par = par[34]
    Henrys_c = par[35]
    n_imp = par[36]
    r = par[37]
    r_imp = par[38]
    Po = par[39]
    epsilon = par[40]
    g = par[41]
    R = par[42]
    X_crit_DO2 = par[43]
    P_crit_DO2 = par[44]
    A_inhib = par[45]
    Tf = par[46]
    Tw = par[47]
    Tcin = par[48]
    Th = par[49]
    Tair = par[50]
    C_ps = par[51]
    C_pw = par[52]
    dealta_H_evap = par[53]
    U_jacket = par[54]
    A_c = par[55]
    Eg = par[56]
    Ed = par[57]
    k_g = par[58]
    k_d = par[59]
    Y_QX = par[60]
    abc = par[61]
    gamma1 = par[62]
    gamma2 = par[63]
    m_ph = par[64]
    K1 = par[65]
    K2 = par[66]
    N_conc_oil = par[67]
    N_conc_paa = par[68]
    N_conc_shot = par[69]
    Y_NX = par[70]
    Y_NP = par[71]
    m_N = par[72]
    X_crit_N = par[73]
    PAA_c = par[74]
    Y_PAA_P = par[75]
    Y_PAA_X = par[76]
    m_PAA = par[77]
    X_crit_PAA = par[78]
    P_crit_PAA = par[79]
    B_1 = par[80]
    B_2 = par[81]
    B_3 = par[82]
    B_4 = par[83]
    B_5 = par[84]
    delta_c_0 = par[85]
    k3 = par[86]
    k1 = par[87]
    k2 = par[88]
    t1 = par[89]
    t2 = par[90]
    q_co2 = par[91]
    X_crit_CO2 = par[92]
    alpha_evp = par[93]
    beta_T = par[94]
    pho_g = par[95]
    pho_oil = par[96]
    pho_w = par[97]
    pho_paa = par[98]
    O_2_in = par[99]
    N2_in = par[100]
    C_CO2_in = par[101]
    Tv = par[102]
    T0 = par[103]
    alpha_1 = par[104]

    # --- 2. Process Inputs Unpacking (inp1) ---
    inhib_flag = inp1[0]
    Fs = inp1[1]
    Fg = inp1[2] / 60.0 # Convert to m^3/s if inp1 is m^3/min? Check units. MATLAB says /60.
    RPM = inp1[3]
    Fc = inp1[4]
    Fh = inp1[5]
    Fb = inp1[6]
    Fa = inp1[7]
    step1 = inp1[8] # ode step size [h]
    Fw = inp1[9]
    if Fw < 0: Fw = 0
    pressure = inp1[10]
    
    # Viscosity flag
    if inp1[25] == 0: # 0 - Uses simulated viscosity
        viscosity = y[9] # y(10) in MATLAB
    else: # 1 - Uses recorded viscosity
        viscosity = inp1[11]

    F_discharge = inp1[12]
    Fpaa = inp1[13]
    Foil = inp1[14]
    NH3_shots = inp1[15]
    dist_flag = inp1[16]
    
    # Disturbances
    distMuP = inp1[17]
    distMuX = inp1[18]
    distsc = inp1[19]
    distcoil = inp1[20]
    distabc = inp1[21]
    distPAA = inp1[22]
    distTcin = inp1[23]
    distO_2_in = inp1[24]

    # Broth viscosity proxy calculation
    # y indices: 3->P, 11->A0, 12->A1, 13->A3, 14->A4 (MATLAB 4, 12, 13, 14, 15)
    pho_b = 1100 + y[3] + y[11] + y[12] + y[13] + y[14]

    # Apply disturbances
    if dist_flag == 1:
        mu_p += distMuP
        mux_max += distMuX
        c_s += distsc
        c_oil += distcoil
        abc += distabc
        PAA_c += distPAA
        Tcin += distTcin
        O_2_in += distO_2_in

    # --- 3. Process Parameters Calculation ---
    
    # Age-dependant term
    # y(11) -> Y(11) Integral, y(12)..y(15) -> biomass regions
    # MATLAB: A_t1 = ((y(11))/(y(12)+y(13)+y(14)+y(15)))
    denom_biomass = y[11] + y[12] + y[13] + y[14]
    if denom_biomass == 0: denom_biomass = 1e-9 # Protect division by zero
    A_t1 = y[10] / denom_biomass # y[10] is Integral of biomass regions (Y(11))

    # Variables mapping
    s = y[0]   # Substrate
    a_1 = y[12] # A1 (Extension) - MATLAB Y(13)
    a_0 = y[11] # A0 (Branching) - MATLAB Y(12)
    a_3 = y[13] # A3 - MATLAB Y(14)
    total_X = denom_biomass # Sum of A0+A1+A3+A4

    # Liquid height
    # y[4] is Volume (Y(5))
    h_b = (y[4] / 1000.0) / (np.pi * (r**2))
    h_b = h_b * (1 - epsilon) # ungassed height

    # Log mean pressure
    # 9.81 * 10^-5 converts Pa to bar approx?
    pressure_bottom = 1 + pressure + ((pho_b * h_b) * 9.81 * 1e-5)
    pressure_top = 1 + pressure
    
    # Avoid log(1) division by zero
    if abs(pressure_bottom - pressure_top) < 1e-6:
        log_mean_pressure = pressure_top
    else:
        log_mean_pressure = (pressure_bottom - pressure_top) / np.log(pressure_bottom / pressure_top)
    
    total_pressure = log_mean_pressure

    # Viscosity min value
    if viscosity < 4:
        viscosity = 1

    # Henry's constant logic check
    DOstar_tp = ((total_pressure) * O_2_in) / Henrys_c

    # --- 4. Inhibition Flags ---
    
    # Initialization
    pH_inhib = 1
    NH3_inhib = 1
    T_inhib = 1
    # mu_h default already set in par[19] but reset in flag 0
    DO_2_inhib_X = 1
    DO_2_inhib_P = 1
    CO2_inhib = 1
    PAA_inhib_X = 1
    PAA_inhib_P = 1

    # Flag 0
    if inhib_flag == 0:
        mu_h = 0.003
        
    # Flag 1
    elif inhib_flag == 1:
        # pH effect
        pH_inhib = 1 / (1 + (y[6]/K1) + (K2/y[6])) # y[6] is pH (Y(7))
        
        # Temperature
        term_g = k_g * np.exp(-(Eg / (R * y[7]))) # y[7] is T (Y(8))
        term_d = k_d * np.exp(-(Ed / (R * y[7])))
        T_inhib = (term_g - term_d) * 0 + 1 # Multiplied by 0? Copied from MATLAB source.
        
        # DO2
        # y[1] is DO2 (Y(2))
        term_do2_x = A_inhib * (X_crit_DO2 * (((total_pressure) * O_2_in) / Henrys_c) - y[1])
        DO_2_inhib_X = 0.5 * (1 - np.tanh(term_do2_x))
        
        term_do2_p = A_inhib * (P_crit_DO2 * (((total_pressure) * O_2_in) / Henrys_c) - y[1])
        DO_2_inhib_P = 0.5 * (1 - np.tanh(term_do2_p))
        
        # pH/Temp effect on hydrolysis
        pH_val = -np.log10(y[6] + 1e-10) # y[6] is H+ concentration ? Note says "Y(7) - X.pH". 
        # Check inputs: indpensim.m converts pH to 10^(-pH). So y[6] is H+ conc.
        # But MATLAB code line: pH = -log10(y(7));
        k4 = np.exp((B_1 + B_2*pH_val + B_3*y[7] + B_4*(pH_val**2)) + B_5*(y[7]**2))
        mu_h = k4

    # Flag 2
    elif inhib_flag == 2:
        pH_inhib = 1 / (1 + (y[6]/K1) + (K2/y[6]))
        
        # Ammonia (y[30] -> Y(31))
        NH3_inhib = 0.5 * (1 - np.tanh(A_inhib * (X_crit_N - y[30])))
        
        # Temperature
        T_inhib = k_g * np.exp(-(Eg / (R * y[7]))) - k_d * np.exp(-(Ed / (R * y[7])))
        
        # CO2 (y[28] -> Y(29) Dissolved CO2)
        CO2_inhib = 0.5 * (1 + np.tanh(A_inhib * (X_crit_CO2 - y[28] * 1000)))
        
        # DO2
        term_do2_x = A_inhib * (X_crit_DO2 * (((total_pressure) * O_2_in) / Henrys_c) - y[1])
        DO_2_inhib_X = 0.5 * (1 - np.tanh(term_do2_x))
        
        term_do2_p = A_inhib * (P_crit_DO2 * (((total_pressure) * O_2_in) / Henrys_c) - y[1])
        DO_2_inhib_P = 0.5 * (1 - np.tanh(term_do2_p))
        
        # PAA (y[29] -> Y(30))
        PAA_inhib_X = 0.5 * (1 + np.tanh(X_crit_PAA - y[29]))
        PAA_inhib_P = 0.5 * (1 + np.tanh(-P_crit_PAA + y[29]))
        
        # Hydrolysis
        pH_val = -np.log10(y[6] + 1e-10)
        k4 = np.exp((B_1 + B_2*pH_val + B_3*y[7] + B_4*(pH_val**2)) + B_5*(y[7]**2))
        mu_h = k4

    # --- 5. Main Rate Equations ---
    
    # Penicillin inhibition curve
    # term: -0.5 * ((s - mean_P)/P_std_dev)^2
    s_term = -0.5 * ((s - mean_P) / P_std_dev)**2
    P_inhib = 2.5 * P_std_dev * ((P_std_dev * np.sqrt(2 * np.pi))**(-1) * np.exp(s_term))

    # Specific growth rates
    mu_a0_calc = ratio_mu_e_mu_b * mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X
    mu_e_calc = mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X

    K_diff_curr = par[10] - (A_t1 * beta_1)
    if K_diff_curr < K_diff_L:
        K_diff_curr = K_diff_L

    # Growing A0
    r_b0 = mu_a0_calc * a_1 * s / (K_b + s)
    r_sb0 = Y_sX * r_b0

    # Non-growing A1
    r_e1 = (mu_e_calc * a_0 * s) / (K_e + s)
    r_se1 = Y_sX * r_e1

    # Differentiation
    r_d1 = mu_diff * a_0 / (K_diff_curr + s)
    r_m0 = m_s * a_0 / (K_diff_curr + s)

    # Vacuoles
    # MATLAB loop: n=17 (Y(17)). In Python y[16] is Y(17).
    # MATLAB phi(1) = y(27). Python phi list.
    phi = []
    # y[26] corresponds to Y(27) dphi_0_dt state ? No, Y(27) is dphi_0_dt integral -> phi_0
    phi.append(y[26]) # k=1
    
    n_idx = 16 # Start index for n (y[16])
    r_mean = []
    
    for k in range(2, 11): # 2 to 10
        r_mean_val = (1.5e-4) + (k - 2) * delta_r
        # y[n_idx] is n_0, n_1... 
        phi_val = ((4 * np.pi * r_mean_val**3) / 3) * y[n_idx] * delta_r
        phi.append(phi_val)
        n_idx += 1
    
    v_2 = sum(phi)
    
    # Density of non-growing regions
    rho_a1 = (a_1 / ((a_1 / rho_a0) + v_2))
    v_a1 = a_1 / (2 * rho_a1) - v_2
    
    # Penicillin production
    # y[3] is P (Y(4))
    r_p = mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P - mu_h * y[3]

    # Vacuole formation
    r_m1 = (m_s * rho_a0 * v_a1 * s) / (K_v + s)

    # Vacuole degeneration
    r_d4 = mu_a * a_3

    # --- 6. Vacuole Dynamics Derivatives ---
    
    # dn0_dt
    dn0_dt = ((mu_v * v_a1) / (K_v + s)) * ((6 / np.pi) * ((r_0 + delta_0)**-3)) - k_v * y[15] # y[15] is n0 (Y(16))

    # dn1_dt to dn9_dt
    # MATLAB: n=17 -> Y(17) corresponds to y[16] (n1)
    # y[15] is n0, y[16] is n1 ... y[24] is n9
    dn_dt_list = []
    
    # We iterate for n1 to n9.
    # Central difference logic: (y(n+1) - y(n-1))
    # Indices in Python:
    # n0 -> 15
    # n1 -> 16
    # ...
    # n9 -> 24
    # nm -> 25
    
    current_idx = 16 # Pointing to n1
    for i in range(1, 10): # 1 to 9
        prev_n = y[current_idx - 1]
        curr_n = y[current_idx]
        next_n = y[current_idx + 1] # Ensure y has enough size. y[25] is nm, valid.
        
        term1 = -k_v * ((next_n - prev_n) / (2 * delta_r))
        term2 = D * (next_n - 2 * curr_n + prev_n) / (delta_r**2)
        dn_dt_list.append(term1 + term2)
        current_idx += 1
    
    # Max vacuole volume department
    n_k = dn_dt_list[-1] # dn9_dt value? No, code says n_k = dn9_dt (value of derivative?)
    # Wait, MATLAB code:
    # dn9_dt = ...
    # n_k = dn9_dt; (This assigns the derivative value to n_k?)
    # Then:
    # k=12; r_m = ...
    # dn_m_dt = k_v*n_k/(r_m-r_k) - mu_a*y(26);
    
    # BUT typically boundary conditions use the concentration, not the derivative.
    # Let's check lines 270-272 in MATLAB:
    # dn9_dt = ...
    # n_k = dn9_dt; 
    # This looks suspicious physically (flux vs concentration), but I strictly copy the code.
    
    n_k_val = dn_dt_list[-1] # This is dn9_dt
    
    # Logic re-check:
    # MATLAB: n_k = y(25); (Line 276) - Wait, line 276 overrides line 271?
    # Line 271: n_k = dn9_dt;
    # ...
    # Line 276: n_k = y(25); (y(25) is n9)
    # The MATLAB code provided has both. Line 276 overrides Line 271. 
    # I will use y[24] (n9) based on line 276 logic which seems more physically sound for a flux out.
    n_k_val = y[24]

    k_idx = 10
    r_k = r_0 + (k_idx - 2) * delta_r
    k_idx_m = 12
    r_m = r_0 + (k_idx_m - 2) * delta_r
    
    # y[25] is nm (Y(26))
    dn_m_dt = k_v * n_k_val / (r_m - r_k) - mu_a * y[25]

    # Mean vacuole
    # y[15] is n0 (Y(16))
    dphi_0_dt = ((mu_v * v_a1) / (K_v + s)) - k_v * y[15] * (np.pi * (r_0 + delta_0)**3) / 6

    # --- 7. Volume and Weight ---
    
    # y[4] is V, y[7] is T
    F_evp = y[4] * alpha_evp * (np.exp(2.5 * (y[7] - T0) / (Tv - T0)) - 1)
    
    pho_feed = (c_s / 1000.0 * pho_g + (1 - c_s / 1000.0) * pho_w)
    dilution = Fs + Fb + Fa + Fw - F_evp + Fpaa
    
    dV1 = Fs + Fb + Fa + Fw + F_discharge / (pho_b / 1000.0) - F_evp + Fpaa
    
    dWt = (Fs * pho_feed / 1000.0 + 
           pho_oil / 1000.0 * Foil + 
           Fb + Fa + Fw + F_discharge - F_evp + 
           Fpaa * pho_paa / 1000.0)

    # --- 8. Biomass Regions Derivatives ---
    
    # da0_dt (Growing)
    da_0_dt = r_b0 - r_d1 - y[11] * dilution / y[4]
    
    # da1_dt (Non-growing)
    # y[12] is a1
    da_1_dt = r_e1 - r_b0 + r_d1 - (np.pi * ((r_k + r_m)**3) / 6) * rho_d * k_v * n_k_val - y[12] * dilution / y[4]
    
    # da3_dt (Degenerated)
    # y[13] is a3
    da_3_dt = (np.pi * ((r_k + r_m)**3) / 6) * rho_d * k_v * n_k_val - r_d4 - y[13] * dilution / y[4]
    
    # da4_dt (Autolysed)
    # y[14] is a4
    da_4_dt = r_d4 - y[14] * dilution / y[4]
    
    # Penicillin P
    dP_dt = r_p - y[3] * dilution / y[4]

    # Active Biomass rate
    X_1 = da_0_dt + da_1_dt + da_3_dt + da_4_dt
    
    # Total biomass
    X_t = y[11] + y[12] + y[13] + y[14]

    # Heat calculations
    Qrxn_X = X_1 * Y_QX * y[4] * Y_O2_X / 1000.0
    Qrxn_P = dP_dt * Y_QX * y[4] * Y_O2_P / 1000.0
    Qrxn_t = Qrxn_X + Qrxn_P
    if Qrxn_t < 0: Qrxn_t = 0
    
    # Power
    N_speed = RPM / 60.0
    D_imp = 2 * r_imp
    unaerated_power = (n_imp * Po * pho_b * (N_speed**3) * (D_imp**5))
    
    # Avoid division by zero if Fg is 0
    if Fg < 1e-9:
        P_g = unaerated_power # Simplified assumption
    else:
        P_g = 0.706 * (((unaerated_power**2) * N_speed * (D_imp**3)) / (Fg**0.56))**0.45
        
    P_n = P_g / unaerated_power if unaerated_power > 0 else 0
    variable_power = (n_imp * Po * pho_b * (N_speed**3) * (D_imp**5) * P_n) / 1000.0

    # --- 9. Final ODE Vector Construction (dy) ---
    
    # Initialize dy vector
    # MATLAB code creates dy up to index 33.
    dy = np.zeros(33)

    # dy[0]: Substrate
    dy[0] = -r_se1 - r_sb0 - r_m0 - r_m1 - ((Y_sP * mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P)) + Fs * c_s / y[4] + Foil * c_oil / y[4] - y[0] * dilution / y[4]

    # dy[1]: Dissolved Oxygen
    V_s = Fg / (np.pi * (r**2))
    # y[7] T, y[4] V
    V_m = y[4] / 1000.0
    P_air = ((V_s * R * y[7] * V_m / (22.4 * h_b)) * np.log(1 + pho_b * 9.81 * h_b / (pressure_top * 10**5)))
    P_t1 = variable_power + P_air
    
    vis_scaled = viscosity / 100.0
    oil_f = Foil / y[4]
    
    kla = (alpha_kla_par * (((V_s**(a_par)) * ((P_t1 / V_m)**b_par) * (vis_scaled)**c_par)) * (1 - oil_f**(d_par)))
    
    OUR = (-X_1) * Y_O2_X - m_O2_X * X_t - dP_dt * Y_O2_P
    OTR = kla * (DOstar_tp - y[1])
    dy[1] = OUR + OTR - (y[1] * dilution / y[4])

    # dy[2]: O2 off-gas
    Vg = epsilon * V_m
    Qfg_in = 60 * Fg * 1000 * 32 / 22.4
    # y[2] is O2 off gas (Y(3)), y[27] is CO2 off gas (Y(28))
    denom_gas = (1 - y[2] - y[27] / 100.0)
    if denom_gas < 1e-6: denom_gas = 1e-6
    
    Qfg_out = 60 * Fg * (N2_in / denom_gas) * 1000 * 32 / 22.4
    dy[2] = (Qfg_in * O_2_in - Qfg_out * y[2] - 0.001 * OTR * V_m * 60) / (Vg * 28.97 * 1000 / 22.4)

    # dy[3]: Penicillin
    dy[3] = dP_dt

    # dy[4]: Volume
    dy[4] = dV1

    # dy[5]: Weight
    dy[5] = dWt

    # dy[6]: pH
    pH_dis = Fs + Foil + Fb + Fa + F_discharge + Fw
    
    if -np.log10(y[6] + 1e-12) < 7: # Acidic
        cb = -abc
        ca = +abc
        pH_balance = 0
    else: # Basic
        cb = +abc
        ca = -abc
        # y(7) update skipped in ODE (handled by solver state update)
        pH_balance = 1
        
    denom_B = (y[4] + Fb * step1 + Fa * step1)
    if denom_B == 0: denom_B = 1e-9
    B_val = (y[6] * y[4] + ca * Fa * step1 + cb * Fb * step1) / denom_B
    B_val = -B_val
    
    term_sqrt = np.sqrt(B_val**2 + 4e-14)
    
    if pH_balance == 1: # Basic
        dy[6] = -gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X) - gamma1 * r_p - gamma2 * (pH_dis) + ((-B_val - term_sqrt) / 2 - y[6])
    else: # Acidic
        dy[6] = +gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X) + gamma1 * r_p + gamma2 * (pH_dis) + ((-B_val + term_sqrt) / 2 - y[6])

    # dy[7]: Temperature
    Ws = P_t1
    Qcon = U_jacket * A_c * (y[7] - Tair)
    
    # Heat transfer terms
    denom_Fc = (Fc / 1000.0 + (alpha_1 * (Fc / 1000.0)**beta_T) / 2 * pho_b * C_ps)
    if denom_Fc == 0: denom_Fc = 1.0
    
    denom_Fh = (Fh / 1000.0 + (alpha_1 * (Fh / 1000.0)**beta_T) / 2 * pho_b * C_ps)
    if denom_Fh == 0: denom_Fh = 1.0

    dQ_dt = (Fs * pho_feed * C_ps * (Tf - y[7]) / 1000.0 + 
             Fw * pho_w * C_pw * (Tw - y[7]) / 1000.0 - 
             F_evp * pho_b * C_pw / 1000.0 - 
             dealta_H_evap * F_evp * pho_w / 1000.0 + 
             Qrxn_t + Ws - 
             (alpha_1 / 1000.0) * Fc**(beta_T + 1) * ((y[7] - Tcin) / denom_Fc) - 
             (alpha_1 / 1000.0) * Fh**(beta_T + 1) * ((y[7] - Th) / denom_Fh) - 
             Qcon)
             
    dy[7] = dQ_dt / ((y[4] / 1000.0) * C_pw * pho_b)

    # dy[8]: Heat generation (Q)
    dy[8] = dQ_dt

    # dy[9]: Viscosity
    # y[9] is Viscosity
    term_k1 = 1 / (1 + np.exp(-k1 * (t - t1)))
    term_k2 = 1 / (1 + np.exp(-k2 * (t - t2)))
    dy[9] = (((3 * (a_0**(1/3)) * term_k1 * term_k2 - k3 * Fw)))

    # dy[10]: Total X (Integral)
    dy[10] = (y[11] + y[12] + y[13] + y[14])

    # dy[11] to dy[14]: Biomass
    dy[11] = da_0_dt
    dy[12] = da_1_dt
    dy[13] = da_3_dt
    dy[14] = da_4_dt

    # dy[15] to dy[24]: Vacuoles
    dy[15] = dn0_dt
    # Fill n1 to n9
    for i, val in enumerate(dn_dt_list):
        dy[16 + i] = val
        
    # dy[25]: Max vacuole volume dept
    dy[25] = dn_m_dt

    # dy[26]: Mean vacuole volume
    dy[26] = dphi_0_dt

    # dy[27]: CO2 off-gas
    total_X_CO2 = y[11] + y[12]
    CER = total_X_CO2 * q_co2 * y[4]
    
    term_co2_in = ((60 * Fg * 44 * 1000) / 22.4) * C_CO2_in
    term_co2_out = ((60 * Fg * 44 * 1000) / 22.4) * y[27]
    
    dy[27] = (term_co2_in + CER - term_co2_out) / (Vg * 28.97 * 1000 / 22.4)

    # dy[28]: Dissolved CO2
    Henrys_c_co2 = (np.exp(11.25 - 395.9 / (y[7] - 175.9))) / (44 * 100)
    C_star_CO2 = (total_pressure * y[27]) / Henrys_c_co2
    dy[28] = kla * delta_c_0 * (C_star_CO2 - y[28]) - y[28] * dilution / y[4]

    # dy[29]: PAA
    dy[29] = Fpaa * PAA_c / y[4] - (Y_PAA_P * dP_dt) - Y_PAA_X * X_1 - m_PAA * y[3] - y[29] * dilution / y[4]

    # dy[30]: Nitrogen (NH3)
    X_C_nitrogen = (-r_b0 - r_e1 - r_d1 - r_d4) * Y_NX
    P_C_nitrogen = -dP_dt * Y_NP
    
    dy[30] = ((NH3_shots * N_conc_shot) / y[4] + 
              X_C_nitrogen + P_C_nitrogen - m_N * total_X + 
              (1 * N_conc_paa * Fpaa / y[4]) + 
              N_conc_oil * Foil / y[4] - 
              y[30] * dilution / y[4])

    # dy[31]: mu_p (stored as state)
    dy[31] = mu_p

    # dy[32]: mu_e (stored as state)
    dy[32] = mu_e_calc

    return dy