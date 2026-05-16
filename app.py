
import streamlit as st
import pandas as pd
import joblib

from utils import (
    load_data,
    show_data,
    run_eda,
    save_model,
    load_model
)

from clustering import perform_clustering
from eda import (
    plot_missing_values,
    plot_correlation_heatmap,
    plot_distributions
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Buyer Segmentation Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🏡 Buyer Segmentation in Real Estate")
st.markdown("AI-Based Investment Profiling & Customer Segmentation Dashboard")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_datasets():
    clients, properties = load_data()
    return clients, properties

clients_df, properties_df = load_datasets()

# ---------------------------------------------------
# DISPLAY DATA INFO
# ---------------------------------------------------
st.subheader("Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.write("### Clients Dataset")
    st.write(clients_df.head())
    st.write("Shape:", clients_df.shape)

with col2:
    st.write("### Properties Dataset")
    st.write(properties_df.head())
    st.write("Shape:", properties_df.shape)

# ---------------------------------------------------
# MERGE DATASETS
# ---------------------------------------------------
if 'client_id' in common_columns: 
    full_df = clients_df.merge( 
        properties_df, on='client_id', how='left' ) 
    st.success("Datasets merged successfully using client_id!") 
else: 
    st.warning( "client_id column not found in both datasets. " "Using clients dataset only." ) 
    full_df = clients_df.copy()
    
# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("🔍 Filters")

country_filter = st.sidebar.multiselect(
    "Country",
    options=full_df['country'].dropna().unique()
)

region_filter = st.sidebar.multiselect(
    "Region",
    options=full_df['region'].dropna().unique()
)

acquisition_filter = st.sidebar.multiselect(
    "Acquisition Purpose",
    options=full_df['acquisition_purpose'].dropna().unique()
)

client_type_filter = st.sidebar.multiselect(
    "Client Type",
    options=full_df['client_type'].dropna().unique()
)

# ---------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------
filtered_df = full_df.copy()

if country_filter:
    filtered_df = filtered_df[
        filtered_df['country'].isin(country_filter)
    ]

if region_filter:
    filtered_df = filtered_df[
        filtered_df['region'].isin(region_filter)
    ]

if acquisition_filter:
    filtered_df = filtered_df[
        filtered_df['acquisition_purpose'].isin(acquisition_filter)
    ]

if client_type_filter:
    filtered_df = filtered_df[
        filtered_df['client_type'].isin(client_type_filter)
    ]

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------
st.subheader("📄 Filtered Data Preview")
show_data(filtered_df)

# ---------------------------------------------------
# KPI METRICS
# ---------------------------------------------------
st.subheader("📊 Dashboard Metrics")

k1, k2, k3 = st.columns(3)

with k1:
    st.metric("Total Buyers", len(filtered_df))

with k2:
    st.metric(
        "Countries",
        filtered_df['country'].nunique()
    )

with k3:
    st.metric(
        "Regions",
        filtered_df['region'].nunique()
    )

# ---------------------------------------------------
# EDA SECTION
# ---------------------------------------------------
st.subheader("📈 Exploratory Data Analysis")

if st.button("Run EDA"):

    st.write("### Missing Values")
    plot_missing_values(filtered_df)

    st.write("### Correlation Heatmap")
    plot_correlation_heatmap(filtered_df)

    st.write("### Distributions")
    plot_distributions(filtered_df)

    st.success("EDA Completed Successfully!")

# ---------------------------------------------------
# CLUSTERING SECTION
# ---------------------------------------------------
st.subheader("🤖 Clustering Analysis")

if st.button("Perform Clustering"):

    with st.spinner("Running clustering model..."):

        model, labels, pca_result = perform_clustering(filtered_df)

        # Save model
        save_model(model, filename='model.pkl')

        st.success("Clustering Completed!")

        # Add cluster labels
        filtered_df['Cluster'] = labels

        # Cluster Distribution
        st.write("### Cluster Distribution")
        st.bar_chart(filtered_df['Cluster'].value_counts())

        # PCA Visualization
        st.write("### PCA Cluster Visualization")

        cluster_plot = perform_clustering(
            filtered_df,
            plot=True
        )

        st.plotly_chart(cluster_plot)

# ---------------------------------------------------
# LOAD SAVED MODEL
# ---------------------------------------------------
st.subheader("💾 Saved Model")

if st.checkbox("Load Saved Model"):

    try:
        model = load_model('model.pkl')
        st.success("Model loaded successfully!")
        st.write(model)

    except Exception as e:
        st.error(f"Error loading model: {e}")

# ---------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------
st.sidebar.header("⬇ Download Report")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name='buyer_segmentation_report.csv',
    mime='text/csv'
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.write(
    "Built with Streamlit | Machine Learning Buyer Segmentation Project"
)


      
