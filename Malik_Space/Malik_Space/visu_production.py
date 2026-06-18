import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Chargement des données en cours...")
df = pd.read_csv('IndPenSim_100_Batches.csv')

# 1. Sécurité : on force la colonne en nombre décimal
df['Batch ID'] = df['Batch ID'].astype(float)

# 2. On récupère tous les IDs de batchs uniques, et on tranche pour garder les 10 premiers
batches_disponibles = df['Batch ID'].unique()
dix_premiers_ids = batches_disponibles[61:71]
print(f"Les 10 batchs sélectionnés sont : {dix_premiers_ids}")

# 3. On filtre le dataset géant pour ne garder que les lignes de ces 10 batchs
df_10_premiers = df[df['Batch ID'].isin(dix_premiers_ids)]

# 4. Création du graphique comparatif
plt.figure(figsize=(14, 7))

# Le paramètre 'hue' sépare automatiquement les courbes par Batch avec la palette de 10 couleurs (tab10)
sns.lineplot(data=df_10_premiers, x='Time (h)', y='Penicillin concentration(P:g/L)', hue='Batch ID', palette='tab10')

plt.title('Comparaison de la Production de Pénicilline (10 Premiers Batchs)')
plt.xlabel('Temps (heures)')
plt.ylabel('Concentration (g/L)')
plt.grid(True)

# Sauvegarde de l'image
plt.savefig('penicilline_10_batchs.png')
print("Graphique comparatif généré avec succès sous le nom 'penicilline_10_batchs.png' !")