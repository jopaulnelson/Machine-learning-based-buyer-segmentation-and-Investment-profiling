import pandas as pd
import joblib
from utils import run_clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_data():
    clients = pd.read_csv('clients.csv')
    properties = pd.read_csv('properties.csv')
    return clients, properties

def show_data(df):
    st.write(df.head(10))
    st.write(f"Shape: {df.shape}")

def run_clustering(data, n_clusters=4):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    # Optional: reduce dimensions for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    return kmeans, labels, pca_result

def run_eda(df):
    from eda import plot_missing_values, plot_correlation_heatmap, plot_distributions, buyer_analysis
    plot_missing_values(df)
    plot_correlation_heatmap(df)
    plot_distributions(df)
    buyer_analysis(df)

def perform_clustering(df):
    from clustering import perform_clustering
    model, labels, pca_result = perform_clustering(df)
    return model, labels, pca_result

def save_model(clustering_model,buyer_segmentation_model_pkl):
    joblib.dump(clustering_model,buyer_segmentation_model_pkl)

def load_model(buyer_segmentation_model_pkl):
    return joblib.load(buyer_segmentation_model_pkl)
      
