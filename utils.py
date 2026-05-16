import pandas as pd
import joblib

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

def perform_clustering(df):
    from clustering import perform_clustering
    model, labels, pca_result = perform_clustering(df)
    return model, labels, pca_result

def save_model(clustering_model,buyer_segmentation_model_pkl):
    joblib.dump(clustering_model,buyer_segmentation_model_pkl)

def load_model(buyer_segmentation_model_pkl):
    return joblib.load(buyer_segmentation_model_pkl)
      
