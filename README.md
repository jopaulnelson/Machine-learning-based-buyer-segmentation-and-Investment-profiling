Buyer Segmentation & Investment Profiling for Real Estate Market Intelligence
This project leverages machine learning techniques to identify distinct customer segments and investment behaviors within the real estate market. The goal is to enable smarter marketing strategies and data-driven investment decisions by uncovering hidden patterns in buyer data.

Background & Context
Real estate markets are characterized by diverse buyer behaviors, including individual homebuyers, institutional investors, international buyers, high-net-worth investors, and first-time buyers. Without proper segmentation, companies often face challenges such as inefficient marketing, poor customer targeting, and missed investment opportunities.
By applying AI-driven clustering algorithms, this project aims to discover underlying customer segments, providing valuable insights into buyer motivations, geographic preferences, and financial behaviors.

Problem Statement
Currently, Parcl lacks a data-driven understanding of:

Different types of property buyers
Investment motivations across demographics
Geographic differences in investment behavior
Customer financing patterns

This gap hampers effective marketing and personalized service offerings.

Data Description
The dataset comprises property transaction records, including features such as:

listing_id, tower_number, transaction_date, unit_category, unit_number, floor_area_sqft, sale_price, listing_status, client_ref, and more.

The dataset enables analysis of buyer behavior, transaction patterns, and property details to support clustering.

Methodology
The project follows a structured data science approach:

Data Cleaning
Handle missing attributes
Remove duplicates


Feature Encoding
Convert categorical variables using One-Hot and Label Encoding


Feature Scaling
Normalize numerical features like age and satisfaction scores


Clustering Algorithms
K-Means for efficiency
Hierarchical clustering for nested insights


Optimal Cluster Selection
Use Elbow Method and Silhouette Score


Cluster Interpretation
Analyze each cluster based on investment purpose, demographics, geographic distribution, and loan behavior




Implementation
The project includes the following scripts:

eda.py: Performs exploratory data analysis and visualization
preprocessing.py: Cleans and prepares the dataset
clustering.py: Implements clustering algorithms and evaluates the optimal number of clusters
utils.py: Contains utility functions for data processing
app.py: Launches a Streamlit web app for interactive analysis


Usage Instructions
Prerequisites

Python 3.8+
Dependencies listed in requirements.txt

Setup

Clone the repository:


          
            
            
          
          git clone https://github.com/yourusername/real-estate-buyer-segmentation.git
cd real-estate-buyer-segmentation
      
Create a virtual environment (optional):


          
            
            
          
          python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
      
Install dependencies:


          
            
            
          
          pip install -r requirements.txt
      
Place your dataset files (e.g., properties.csv) in the project directory or update the dataset paths in scripts.

Running the Analysis

Run exploratory data analysis:


          
            
            
          
          python eda.py
      
Run clustering analysis:


          
            
            
          
          python clustering.py
      
Launch the Streamlit dashboard:


          
            
            
          
          streamlit run app.py
      Open your browser to http://localhost:8501 to explore the interactive insights.

Results & Insights
The analysis reveals:

Key customer segments such as global investors, first-time buyers, corporate buyers, and luxury investors
Investment patterns by region and buyer type
Demographic and financial characteristics of each segment

These insights help tailor marketing efforts and identify promising investment opportunities.

Future Work & Enhancements

Incorporate additional data sources for richer segmentation
Utilize advanced clustering techniques (e.g., DBSCAN, Gaussian Mixture)
Develop real-time dashboards for dynamic analytics
Automate model retraining with new data
