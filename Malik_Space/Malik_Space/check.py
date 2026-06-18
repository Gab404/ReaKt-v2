import pandas as pd

print("⏳ Chargement du fichier CSV pour audit Raman...")
df = pd.read_csv('IndPenSim_100_Batches.csv')

# Extraction du bloc Raman (tout ce qui se trouve après la 39ème colonne)
colonnes_raman = df.columns[39:]

print("\n=======================================================")
print(" 🔬 AUDIT DE QUALITÉ : SPECTRES RAMAN")
print("=======================================================\n")

# 1. Vérification des dimensions et des longueurs d'onde
print("👉 TEST 1 : DIMENSIONS ET PLAGE SPECTRALE")
print(f"Nombre total de variables (wavenumbers) par spectre : {len(colonnes_raman)}")
print(f"Première colonne (Wavenumber min) : '{colonnes_raman[0]}'")
print(f"Dernière colonne (Wavenumber max)  : '{colonnes_raman[-1]}'")
print("Note : Selon la publication, on devrait être autour de 250 à 2250 cm^-1.")

# 2. Vérification de l'intégrité (recherche de trous de données)
print("\n👉 TEST 2 : INTÉGRITÉ DES CAPTEURS (NaN)")
trous_raman = df[colonnes_raman].isna().sum().sum()
lignes_totales = len(df)
cellules_totales = lignes_totales * len(colonnes_raman)
pourcentage_vide = (trous_raman / cellules_totales) * 100

print(f"Total des cases dans le bloc Raman : {cellules_totales}")
print(f"Nombre de cases vides (NaN)        : {trous_raman} ({pourcentage_vide:.2f}%)")

if trous_raman > 0:
    print("⚠️ ATTENTION : Le capteur Raman a des trous de données ! Notre '.fillna(0)' précédent était donc bien indispensable pour éviter un crash du modèle.")
else:
    print("✅ Le capteur Raman a fonctionné en continu sans aucune perte de signal.")

# 3. Échantillon visuel du tout premier spectre valide
print("\n👉 TEST 3 : LECTURE DU PREMIER SPECTRE")
print("Voici un extrait des intensités lumineuses mesurées à l'Heure 0.2 :")
premier_spectre = df[colonnes_raman].iloc[0]

# Affichage des 5 premières et 5 dernières valeurs du spectre
print("Début du spectre :")
print(premier_spectre.head(5).to_string())
print("...")
print("Fin du spectre :")
print(premier_spectre.tail(5).to_string())