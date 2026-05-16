import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def handle_missing_data(df):
    """
    Fill missing values for categorical and numeric columns.
    """
    categorical_cols = ['client_type', 'gender', 'country', 'region', 'acquisition_purpose', 'referral_channel']
    numeric_cols = ['satisfaction_score', 'date_of_birth']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df

def remove_duplicates(df):
    """
    Remove duplicate client entries based on 'client_id'.
    """
    return df.drop_duplicates(subset='client_id')

def encode_categorical(df, categorical_cols):
    """
    One-Hot Encode categorical features.
    """
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    df.drop(columns=categorical_cols, inplace=True)
    return df

def scale_numeric(df, numeric_cols):
    """
    Normalize numeric features using StandardScaler.
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def run_clustering(data, n_clusters=4):
    """
    Run KMeans clustering and PCA for visualization.
    """
    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    return kmeans, labels, pca_result

def save_model(clustering_model, filename):
    """
    Save clustering model to disk.
    """
    joblib.dump(clustering_model, filename)

def load_model(filename):
    """
    Load clustering model from disk.
    """
    return joblib.load(filename)
      
