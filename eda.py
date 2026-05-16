import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

def plot_missing_values(df):
    plt.figure(figsize=(10,4))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title('Missing Values Heatmap')
    st.pyplot()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot()

def plot_distributions(df):
    for col in ['floor_area_sqft', 'sale_price', 'satisfaction_score', 'age']:
        st.subheader(f'Distribution of {col}')
        fig = px.histogram(df, x=col, nbins=30)
        st.plotly_chart(fig)

def buyer_analysis(df):
    st.subheader("Country-wise Buyer Analysis")
    country_counts = df['country'].value_counts()
    st.bar_chart(country_counts)

    st.subheader("Region-wise Investment Analysis")
    region_counts = df['region'].value_counts()
    st.bar_chart(region_counts)

    st.subheader("Loan Dependency")
    loan_counts = df['loan_applied'].value_counts()
    st.bar_chart(loan_counts)

    st.subheader("Customer Satisfaction")
    satisfaction_counts = df['satisfaction_score'].value_counts()
    st.bar_chart(satisfaction_counts)

    st.subheader("Referral Channel Analysis")
    referral_counts = df['referral_channel'].value_counts()
    st.bar_chart(referral_counts)
