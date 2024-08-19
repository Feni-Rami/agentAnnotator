import pandas as pd # type: ignore

# Charger les données générées
try:
    data_df = pd.read_csv('data/medical_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("Le fichier data/medical_data.csv est introuvable.")

# Valider la présence de la colonne 'truth'
if 'truth' not in data_df.columns:
    raise ValueError("La colonne 'truth' est manquante dans le fichier de données.")

# Extraire les caractéristiques et les étiquettes
data = data_df.drop(columns=['truth']).values
truth = data_df['truth'].values

# Vérifier que les données et les vérités terrain ont la même longueur
if len(data) != len(truth):
    raise ValueError("La longueur des données et des vérités terrain ne correspond pas.")

# Configuration des paramètres
config = {
    "num_features": data.shape[1],
    "num_actions": 2,  # Nombre d'actions possibles
    "num_annotators": 12,  # Nombre d'annotateurs
    "data": data,  # Les données normalisées
    "truth": truth,  # Vérités terrain
    "time_budget_range": (0.5, 1.5),  # Plage de temps alloué pour les annotations
    "confidence_range": (0.1, 0.9),  # Plage de confiance initiale des annotateurs
    "stress_range": (0.3, 0.7),  # Plage de stress initial des annotateurs
    "fatigue_range": (0.4, 0.6),  # Plage de fatigue initiale des annotateurs
    "num_gpus": 0,  # Nombre de GPUs utilisés (0 si aucun)
}
