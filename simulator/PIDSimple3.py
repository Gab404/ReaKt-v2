def pid_simple_3(uk1, ek, ek1, yk, yk1, yk2, u_min, u_max, Kp, Ti, Td, h):
    """
    Contrôleur PID simple sous forme incrémentale.
    La dérivée est appliquée uniquement sur la sortie (PV) pour éviter les à-coups sur consigne.
    
    Args:
        uk1: Commande précédente (Output t-1)
        ek: Erreur actuelle (SP - PV)
        ek1: Erreur précédente
        yk: Valeur mesurée actuelle (PV)
        yk1: Valeur mesurée précédente
        yk2: Valeur mesurée avant-précédente
        u_min: Saturation basse
        u_max: Saturation haute
        Kp: Gain proportionnel
        Ti: Constante de temps intégrale
        Td: Constante de temps dérivée
        h: Pas de temps d'échantillonnage
        
    Returns:
        u: Nouvelle commande saturée
    """
    
    # 1. Composante Proportionnelle (différence d'erreur)
    P = ek - ek1
    
    # 2. Composante Intégrale
    if Ti > 1e-7:
        I = ek * h / Ti
    else:
        I = 0.0
        
    # 3. Composante Dérivée (sur la mesure yk, pas sur l'erreur)
    # Note: Le signe moins est normal car la dérivée s'oppose au changement de PV
    if Td > 0.001:
        D = -Td / h * (yk - 2 * yk1 + yk2)
    else:
        D = 0.0
        
    # 4. Calcul de la nouvelle commande (Forme incrémentale)
    # u_new = u_old + delta_u
    u = uk1 + Kp * (P + I + D)
    
    # 5. Saturation (Anti-windup implicite par la forme incrémentale)
    if u > u_max:
        u = u_max
    if u < u_min:
        u = u_min
        
    return u