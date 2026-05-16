from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import numpy as np

def perform_clustering(df):
    # Prepare data
    features = df.select_dtypes(include='number')
    # Find optimal clusters
    sse = []
    silhouette_scores = []
    k_range = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(features)
        sse.append(km.inertia_)
        score = silhouette_score(features, km.labels_)
        silhouette_scores.append(score)

    # Plot Elbow
    plt.figure()
    plt.plot(k_range, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for KMeans')
    plt.savefig('elbow.png')

    # Choose best k based on silhouette
    best_k = k_range[np.argmax(silhouette_scores)]
    km = KMeans(n_clusters=best_k, random_state=42)
    km.fit(features)

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    return km, km.labels_, pca_result

def plot_clusters(labels, pca=False, pca_data=None):
    if pca and pca_data is not None:
        df_plot = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        df_plot['cluster'] = labels
        fig = px.scatter(df_plot, x='PC1', y='PC2', color='cluster', title='KMeans Clusters (PCA)')
        return fig
    else:
        # Placeholder
        pass

def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)

def load_model(filename='model.pkl'):
    return joblib.load(filename)
