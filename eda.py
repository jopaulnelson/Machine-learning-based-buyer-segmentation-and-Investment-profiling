
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# MISSING VALUES HEATMAP
# ---------------------------------------------------

def plot_missing_values(df):

    if df.empty:
        st.warning("Dataset is empty.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        yticklabels=False,
        ax=ax
    )

    ax.set_title("Missing Values Heatmap")

    st.pyplot(fig)

# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------

def plot_correlation_heatmap(df):

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        st.warning("No numeric columns found.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    corr = numeric_df.corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)

# ---------------------------------------------------
# DISTRIBUTION PLOTS
# ---------------------------------------------------

def plot_distributions(df):

    import plotly.express as px
    import streamlit as st

    # Get numeric columns only
    numeric_cols = df.select_dtypes(
        include=['number']
    ).columns

    if len(numeric_cols) == 0:

        st.warning(
            "No numeric columns available."
        )

        return

    # Create histogram for each numeric column
    for col in numeric_cols:

        st.subheader(f"Distribution of {col}")

        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"Distribution of {col}"
        )

        st.plotly_chart(fig)

# ---------------------------------------------------
# BUYER ANALYSIS
# ---------------------------------------------------

def buyer_analysis(df):

    # Country Analysis
    if 'country' in df.columns:

        st.subheader("Country-wise Buyer Analysis")

        country_counts = df['country'].value_counts()

        st.bar_chart(country_counts)

    # Region Analysis
    if 'region' in df.columns:

        st.subheader("Region-wise Investment Analysis")

        region_counts = df['region'].value_counts()

        st.bar_chart(region_counts)

    # Loan Analysis
    if 'loan_applied' in df.columns:

        st.subheader("Loan Dependency")

        loan_counts = df['loan_applied'].value_counts()

        st.bar_chart(loan_counts)

    # Satisfaction Analysis
    if 'satisfaction_score' in df.columns:

        st.subheader("Customer Satisfaction")

        satisfaction_counts = df['satisfaction_score'].value_counts()

        st.bar_chart(satisfaction_counts)

    # Referral Channel
    if 'referral_channel' in df.columns:

        st.subheader("Referral Channel Analysis")

        referral_counts = df['referral_channel'].value_counts()

        st.bar_chart(referral_counts)

