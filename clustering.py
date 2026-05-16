
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------
# CLUSTERING FUNCTION
# ---------------------------------------------------

def perform_clustering(df):

    # Make copy
    data = df.copy()

    # ---------------------------------------------------
    # DROP UNNECESSARY COLUMNS
    # ---------------------------------------------------

    drop_cols = [
        'client_id',
        'first_name',
        'last_name'
    ]

    for col in drop_cols:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    # ---------------------------------------------------
    # HANDLE DATE OF BIRTH -> AGE
    # ---------------------------------------------------

    if 'date_of_birth' in data.columns:

        data['date_of_birth'] = pd.to_datetime(
            data['date_of_birth'],
            errors='coerce'
        )

        current_year = pd.Timestamp.now().year

        data['age'] = (
            current_year -
            data['date_of_birth'].dt.year
        )

        data.drop(columns=['date_of_birth'], inplace=True)

    # ---------------------------------------------------
    # ENCODE CATEGORICAL COLUMNS
    # ---------------------------------------------------

    categorical_cols = data.select_dtypes(
        include=['object']
    ).columns

    encoder = LabelEncoder()

    for col in categorical_cols:

        data[col] = data[col].astype(str)

        data[col] = encoder.fit_transform(data[col])

    # ---------------------------------------------------
    # HANDLE MISSING VALUES
    # ---------------------------------------------------

    data.fillna(0, inplace=True)

    # ---------------------------------------------------
    # FEATURE SCALING
    # ---------------------------------------------------

    scaler = StandardScaler()

    features = scaler.fit_transform(data)

    # ---------------------------------------------------
    # KMEANS CLUSTERING
    # ---------------------------------------------------

    km = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    km.fit(features)

    # ---------------------------------------------------
    # PCA VISUALIZATION
    # ---------------------------------------------------

    n_features = features.shape[1]

    # PCA requires at least 2 features
    if n_features >= 2:

        pca = PCA(n_components=2)

        pca_result = pca.fit_transform(features)

    else:

        st.warning(
            "Not enough features for PCA visualization."
        )

        pca_result = None

    return km, km.labels_, pca_result


