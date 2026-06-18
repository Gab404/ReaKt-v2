import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =====================================================================
# 1. REDÉFINITION DE L'ARCHITECTURE CNN 1D V4
# =====================================================================
class FusionModel(nn.Module):
    def __init__(self, phys_dim):
        super(FusionModel, self).__init__()
        self.raman_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten()
        )
        self.phys_branch = nn.Sequential(
            nn.Linear(phys_dim, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(528, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x_raman, x_phys):
        feat_raman = self.raman_branch(x_raman)
        feat_phys = self.phys_branch(x_phys)
        combined = torch.cat((feat_raman, feat_phys), dim=1)
        return self.fusion_head(combined)

# =====================================================================
# 2. PRÉPARATION DU TEST SET (LES 20 CUVES INVISIBLES)
# =====================================================================
df = pd.read_csv('IndPenSim_100_Batches.csv')
df = df.rename(columns={'Batch ID': 'Volume (L)'})
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1
df = df.sort_values(by=['VRAI_Batch_ID', 'Time (h)']).reset_index(drop=True)

colonnes_raman = df.columns[39:]
colonnes_physiques = ['pH(pH:pH)', 'Temperature(T:K)', 'Agitator RPM(RPM:RPM)', 'Sugar feed rate(Fs:L/h)']
colonnes_cibles = ['Offline Biomass concentratio(X_offline:X(g L^{-1}))', 'Viscosity(Viscosity_offline:centPoise)']

df[colonnes_cibles] = df.groupby('VRAI_Batch_ID')[colonnes_cibles].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
df_propre = df.dropna(subset=colonnes_cibles, how='any').dropna(subset=colonnes_raman, how='all').fillna(0)

batches_uniques = df_propre['VRAI_Batch_ID'].unique()
random.seed(42)
random.shuffle(batches_uniques)
test_batches = batches_uniques[int(len(batches_uniques) * 0.8):]
df_test = df_propre[df_propre['VRAI_Batch_ID'].isin(test_batches)]

# =====================================================================
# 3. CHARGEMENT ET PRÉPARATION DES TENSEURS
# =====================================================================
scaler_raman = joblib.load('scaler_raman.pkl')
scaler_phys = joblib.load('scaler_phys.pkl')
scaler_y = joblib.load('scaler_y.pkl')

T_raman_test = torch.FloatTensor(scaler_raman.transform(df_test[colonnes_raman].values)).unsqueeze(1)
T_phys_test = torch.FloatTensor(scaler_phys.transform(df_test[colonnes_physiques].values))
T_y_test = torch.FloatTensor(scaler_y.transform(df_test[colonnes_cibles].values))

test_loader = DataLoader(TensorDataset(T_raman_test, T_phys_test, T_y_test), batch_size=256, shuffle=False)

model = FusionModel(phys_dim=4)
# Chargement du meilleur modèle sauvegardé
model.load_state_dict(torch.load('reakt_fusion_v4_best.pth'))
model.eval()

# =====================================================================
# 4. INFERENCE ET CALCUL DES MÉTRIQUES (SANS PRINT)
# =====================================================================
preds_list, vraies_list = [], []

with torch.no_grad():
    for b_ram, b_phy, b_y in test_loader:
        preds = model(b_ram, b_phy)
        preds_list.append(preds.numpy())
        vraies_list.append(b_y.numpy())

preds_reelles = scaler_y.inverse_transform(np.vstack(preds_list))
vraies_reelles = scaler_y.inverse_transform(np.vstack(vraies_list))

y_true_bio, y_pred_bio = vraies_reelles[:, 0], preds_reelles[:, 0]
y_true_visco, y_pred_visco = vraies_reelles[:, 1], preds_reelles[:, 1]

# =====================================================================
# 5. PARITY PLOT (Graphique de Parité avec Métriques Intégrées)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_parity(ax, y_true, y_pred, title, color, unit):
    # Calcul des métriques
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Tracé du nuage de points
    ax.scatter(y_true, y_pred, alpha=0.15, color=color, s=10)
    
    # Ligne idéale Y = X
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Idéal (Y = X)")
    
    # Incrustation de la boîte de texte avec les métriques
    textstr = f"R² = {r2:.3f}\nRMSE = {rmse:.2f} {unit}\nMAE = {mae:.2f} {unit}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Vraie Valeur ({unit})", fontsize=12)
    ax.set_ylabel(f"Prédiction IA ({unit})", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')

plot_parity(ax1, y_true_bio, y_pred_bio, "Parity Plot : Biomasse (CNN 1D V4)", 'blue', 'g/L')
plot_parity(ax2, y_true_visco, y_pred_visco, "Parity Plot : Viscosité (CNN 1D V4)", 'red', 'cP')

plt.tight_layout()
plt.show()