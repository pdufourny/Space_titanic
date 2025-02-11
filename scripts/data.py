import os
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from google.cloud import storage

# Client Google Cloud Storage
storage_client = storage.Client.from_service_account_json("spaceship-key.json")
BUCKET_NAME = 'spaceship_titanic_bucket'
DATA_FOLDER = 'data'  # Dossier pour stocker les fichiers téléchargés et traités

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

# Fonction pour prétraiter les données
def preprocess_data(df, is_train=True, imputer=None):
    """Prétraite le DataFrame d'entrée."""
    
    # Supprimer les colonnes non pertinentes pour la prédiction
    df = df.drop(columns=['PassengerId', 'Name'])
    
    # Traiter les valeurs manquantes (en remplissant avec des valeurs par défaut)
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)
    df['VIP'] = df['VIP'].fillna(False).astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Créer de nouvelles caractéristiques à partir de 'Cabin'
    cabin_df = df['Cabin'].str.split('/', expand=True)
    df['Cabin_Deck'] = cabin_df[0]
    df['Cabin_Num'] = cabin_df[1]
    df['Cabin_Side'] = cabin_df[2]
    df = df.drop(columns=['Cabin'])  # Supprimer la colonne d'origine 'Cabin'
    
    # Traiter les colonnes catégorielles (par exemple, HomePlanet, Destination)
    df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
    df['Destination'] = df['Destination'].fillna('Unknown')
    
    # Encodage one-hot pour les colonnes catégorielles
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side'], drop_first=True)
    
    # Si c'est l'ensemble d'entraînement, on sépare X et y
    if is_train:
        # Vérifier si 'Transported' existe avant de la supprimer
        if 'Transported' in df.columns:
            y = df['Transported']
            X = df.drop(columns=['Transported'])
        else:
            raise KeyError("'Transported' column not found in training data.")
    else:
        # Pour les données de test, il n'y a pas de colonne 'Transported'
        X = df
        y = None
    
    # Si imputer n'est pas passé, l'entraîner
    if is_train:
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
    else:
        # Pour les données de test, utiliser l'imputer déjà formé
        X_imputed = imputer.transform(X)
    
    # Retourner les données transformées et l'imputer
    return X_imputed, y, imputer

# Fonction principale pour télécharger, prétraiter et télécharger les données
def main():
    # Télécharger les données brutes depuis Google Cloud Storage
    download_file_from_gcs(BUCKET_NAME, 'train.csv', f'{DATA_FOLDER}/train.csv')
    download_file_from_gcs(BUCKET_NAME, 'test.csv', f'{DATA_FOLDER}/test.csv')

    # Charger les données dans un DataFrame pandas
    train_df = pd.read_csv(f'{DATA_FOLDER}/train.csv')
    test_df = pd.read_csv(f'{DATA_FOLDER}/test.csv')

    # Prétraiter les données d'entraînement
    X_train_imputed, y_train, imputer = preprocess_data(train_df, is_train=True)

    # Prétraiter les données de test avec l'imputer appris
    X_test_imputed, _, _ = preprocess_data(test_df, is_train=False, imputer=imputer)  # Pas besoin de y pour les données de test

    # Sauvegarder les données prétraitées localement
    pd.DataFrame(X_train_imputed).to_csv(f'{DATA_FOLDER}/X_train_imputed.csv', index=False)
    pd.DataFrame(X_test_imputed).to_csv(f'{DATA_FOLDER}/X_test_imputed.csv', index=False)
    print("Données prétraitées sauvegardées localement.")

    # Télécharger les données prétraitées vers Google Cloud Storage
    upload_file_to_gcs(BUCKET_NAME, f'{DATA_FOLDER}/X_train_imputed.csv', 'X_train_imputed.csv')
    upload_file_to_gcs(BUCKET_NAME, f'{DATA_FOLDER}/X_test_imputed.csv', 'X_test_imputed.csv')

    # Sauvegarder l'imputer pour pouvoir l'utiliser pendant l'inférence
    with open(f'{DATA_FOLDER}/imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    print("Imputer sauvegardé localement.")

    # Télécharger l'imputer sur Google Cloud Storage
    upload_file_to_gcs(BUCKET_NAME, f'{DATA_FOLDER}/imputer.pkl', 'imputer.pkl')

if __name__ == "__main__":
    main()
