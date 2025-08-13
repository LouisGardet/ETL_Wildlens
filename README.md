# 🐾 ETL_Wildlens

Projet d’extraction, transformation et chargement (ETL) pour la classification d’empreintes animales à partir du dataset OpenAnimalTracks.  
Ce projet fait partie d’un travail académique dans le cadre du Bachelor Intelligence Artificielle et Data Science à l’EPSI.

---

## 📁 Structure du projet

ETL_Wildlens/
│
├── data/                     # Données (certaines exclues du dépôt)
│   ├── metadata/             # Fichiers de métadonnées CSV
│   └── processed/            # Données préparées (.npz, index)
│
├── notebooks/                # Notebooks Jupyter pour ETL et ML
│   ├── 1_ETL_data_preparation.ipynb
│   └── 2_ML_model_training.ipynb
│
├── src/                      # Scripts Python
│   └── init.py
│
├── .gitignore
├── README.md
└── main.py

---

## 🚀 Objectifs

- Nettoyer et unifier les métadonnées de plusieurs sources (fusion de CSV)
- Structurer les images et indexer les données pour l’entraînement
- Entraîner un modèle CNN (MobileNetV2) pour la reconnaissance d’espèces animales
- Permettre une réutilisation simple du pipeline

---

## 🛠️ Installation

1. Cloner le dépôt :
```bash
git clone git@github.com:LouisGardet/ETL_Wildlens.git
cd ETL_Wildlens
```

2.	Créer un environnement virtuel et installer les dépendances :
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # à créer si besoin
```

---

📦 Données

⚠️ Les images sources ne sont pas fournies dans ce dépôt pour des raisons de droits d’utilisation.
Seules les métadonnées sont disponibles dans data/metadata/.

---

📚 Contenu des notebooks
	•	1_ETL_data_preparation.ipynb : pipeline de nettoyage, fusion, structuration des fichiers, création de l’index.
	•	2_ML_model_training.ipynb : chargement des données, data augmentation, entraînement du modèle MobileNetV2.

⸻

🤖 Modèle

Le modèle final est un CNN basé sur MobileNetV2, préentraîné sur ImageNet, affiné pour la classification d’empreintes.

⸻

🔒 Licence & crédits
	•	Projet réalisé dans le cadre d’un cursus académique à l’EPSI.
	•	Dataset OpenAnimalTracks : https://www.openanimaltracks.org/
	•	Aucune redistribution des données brutes n’est autorisée.

⸻

✍️ Auteur

Louis Gardet – GitHub
Bachelor IA & Data Science – EPSI Montpellier