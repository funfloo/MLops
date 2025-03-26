import os
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(data, test_size=0.3, random_state=42):
    """
    Sépare le jeu de données en ensembles d'entraînement et de test.

    Paramètres:
        data (pd.DataFrame) : le DataFrame contenant les données.
        test_size (float) : proportion des données utilisées pour le test.
        random_state (int) : graine aléatoire pour la reproductibilité.

    Retourne:
        train_data (pd.DataFrame) : données d'entraînement.
        test_data (pd.DataFrame) : données de test.
    """
    logger.info("Début du split des données avec test_size=%s et random_state=%s", test_size, random_state)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    logger.info("Split terminé: %s échantillons pour l'entraînement et %s pour le test", len(train_data), len(test_data))
    return train_data, test_data

if __name__ == "__main__":
    # Activation de l'autologging MLflow pour suivre paramètres, métriques et modèles
    mlflow.sklearn.autolog()
    
    # Chemin vers le fichier CSV dans le dossier experimentation/data
    # (le chemin ci-dessous est relatif depuis src/model)
    data_file = os.path.join(os.path.dirname(__file__), "..", "..", "experimentation", "data", "ton_fichier.csv")
    
    logger.info("Chargement des données depuis %s", data_file)
    data = pd.read_csv(data_file)
    
    # Si besoin, tu peux ajouter ici des étapes de prétraitement
    # ...

    # Split des données
    train_data, test_data = split_data(data, test_size=0.3, random_state=42)
    
    # Suite du code de training (extraction des features, entraînement du modèle, etc.)
    # Par exemple :
    # X_train = train_data.drop("target", axis=1)
    # y_train = train_data["target"]
    # X_test = test_data.drop("target", axis=1)
    # y_test = test_data["target"]
    # ... Entraînement et évaluation du modèle

    logger.info("Entraînement terminé")