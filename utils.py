import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_data():
    clients = pd.read_csv('clients.csv')
    properties = pd.read_csv('properties.csv')
    return clients, properties

def show_data(df):
    st.write(df.head(10))
    st.write(f"Shape: {df.shape}")

def run_eda(df):
    from eda import plot_missing_values, plot_correlation_heatmap, plot_distributions, buyer_analysis
    plot_missing_values(df)
    plot_correlation_heatmap(df)
    plot_distributions(df)
    buyer_analysis(df)

def plot_clusters(pca_result, labels):
    """
    Plot PCA-reduced data with cluster labels.
    """
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clusters PCA Plot')
    plt.legend(*scatter.legend_elements(), title='Clusters')
    plt.show()
    
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
      
