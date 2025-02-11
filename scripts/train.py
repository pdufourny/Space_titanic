import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier  # Importation de XGBoost
from sklearn.metrics import accuracy_score
from google.cloud import storage

# Client Google Cloud Storage
storage_client = storage.Client.from_service_account_json("spaceship-key.json")
BUCKET_NAME = 'spaceship_titanic_bucket'
DATA_FOLDER = 'data'  # Dossier pour stocker les fichiers téléchargés et traités
MODELS_FOLDER = os.path.join(DATA_FOLDER, 'models')  # Dossier pour sauvegarder les modèles

# Créer le dossier 'models' s'il n'existe pas
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

# Fonction pour télécharger un fichier depuis Google Cloud Storage
def download_file_from_gcs(bucket_name, blob_name, destination_file_name):
    """Télécharge un fichier depuis GCS vers un fichier local."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Téléchargé {blob_name} vers {destination_file_name}")

# Fonction pour télécharger un fichier vers Google Cloud Storage
def upload_file_to_gcs(bucket_name, local_file_path, gcs_blob_name):
    """Télécharge un fichier vers GCS."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"Téléchargé {local_file_path} vers {gcs_blob_name}")

# Fonction pour entraîner un modèle
def train_model(X_train, y_train, model_type='rf'):
    """Entraîne un modèle en fonction du type spécifié."""
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'xgb':  # Utilisation de XGBoost
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    else:
        raise ValueError("Modèle inconnu, choisissez entre 'rf', 'lr' et 'xgb'.")
    
    model.fit(X_train, y_train)
    return model

# Fonction principale pour entraîner et sauvegarder les modèles
def main():
    # Télécharger les données prétraitées depuis Google Cloud Storage
    download_file_from_gcs(BUCKET_NAME, 'X_train_imputed.csv', f'{DATA_FOLDER}/X_train_imputed.csv')
    download_file_from_gcs(BUCKET_NAME, 'X_test_imputed.csv', f'{DATA_FOLDER}/X_test_imputed.csv')
    download_file_from_gcs(BUCKET_NAME, 'train.csv', f'{DATA_FOLDER}/train.csv')

    # Charger les données prétraitées dans des DataFrames pandas
    X_train = pd.read_csv(f'{DATA_FOLDER}/X_train_imputed.csv')
    X_test = pd.read_csv(f'{DATA_FOLDER}/X_test_imputed.csv')
    train_df = pd.read_csv(f'{DATA_FOLDER}/train.csv')

    # Extraire y_train à partir du DataFrame d'entraînement
    y_train = train_df['Transported']

    # Liste des modèles à entraîner
    models_to_train = ['rf', 'lr', 'xgb']  # 'xgb' pour XGBoost

    # Entraîner et évaluer chaque modèle
    for model_type in models_to_train:
        print(f"\nEntraînement et évaluation du modèle {model_type}...")
        
        # Entraîner le modèle
        model = train_model(X_train, y_train, model_type=model_type)

        # Prédire sur les données de test
        y_pred = model.predict(X_test)

        # Évaluer le modèle (ici, accuracy)
        accuracy = accuracy_score(y_train, model.predict(X_train))  # Accuracy sur les données d'entraînement
        print(f"Précision sur les données d'entraînement pour {model_type}: {accuracy:.4f}")

        # Sauvegarder le modèle dans un fichier pickle dans le dossier 'models'
        model_filename = f'{model_type}_model.pkl'  # Nom du fichier dépendant du type de modèle
        model_filepath = os.path.join(MODELS_FOLDER, model_filename)
        
        with open(model_filepath, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Modèle {model_type} sauvegardé localement sous {model_filepath}")

        # Télécharger le modèle sur Google Cloud Storage
        upload_file_to_gcs(BUCKET_NAME, model_filepath, f'models/{model_filename}')

if __name__ == "__main__":
    main()
