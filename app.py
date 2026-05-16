import streamlit as st
from utils import load_data, show_data, run_eda, run_clustering, plot_clusters, save_model, load_model
from clustering import perform_clustering
from eda import plot_missing_values, plot_correlation_heatmap, plot_distributions
import joblib

# Set page config
st.set_page_config(page_title="Buyer Segmentation Dashboard", layout="wide")

# Sidebar filters
st.sidebar.header("Filters")
country_filter = st.sidebar.multiselect("Country", options=None)
region_filter = st.sidebar.multiselect("Region", options=None)
acquisition_filter = st.sidebar.multiselect("Acquisition Purpose", options=None)
client_type_filter = st.sidebar.multiselect("Client Type", options=None)

# Load data
@st.cache
def load_datasets():
    clients, properties = load_data()
    return clients, properties

clients_df, properties_df = load_datasets()
st.write("Clients Data Columns:", clients_df.columns)
st.write("Properties Data Columns:", properties_df.columns)
# Merge datasets on client_id
full_df = clients_df.merge(properties_df, on=client_id, how='left')

# Apply filters
if country_filter:
    full_df = full_df[full_df['country'].isin(country_filter)]
if region_filter:
    full_df = full_df[full_df['region'].isin(region_filter)]
if acquisition_filter:
    full_df = full_df[full_df['acquisition_purpose'].isin(acquisition_filter)]
if client_type_filter:
    full_df = full_df[full_df['client_type'].isin(client_type_filter)]

# Display dataset preview
st.title("Buyer Segmentation in Real Estate")
st.subheader("Data Preview")
show_data(full_df)

# Run EDA
if st.button("Run EDA"):
    run_eda(full_df)

# Run Clustering
if st.button("Perform Clustering"):
    model, labels, pca_result = perform_clustering(full_df)
    save_model(model, filename='model.pkl')
    st.success("Clustering Completed and Model Saved!")
    # Plot cluster distribution
    plot_clusters(labels)
    # Show PCA plot
    st.subheader("PCA of Clusters")
    st.plotly_chart(plot_clusters(labels, pca=True, pca_data=pca_result))

# Load and display saved model
if st.checkbox("Load saved model"):
    model = load_model('model.pkl')
    st.write("Model loaded. You can now use it for predictions.")

# Download option for reports
@st.cache
def generate_report():
    # Assume report generation code here
    return None

st.sidebar.header("Download Reports")
# Implement CSV download button if needed

# Run
st.write("Use the sidebar to filter data, and buttons to run EDA and clustering.")
      
