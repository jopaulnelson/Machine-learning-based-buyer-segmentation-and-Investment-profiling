import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def handle_missing(df):
    # Fill missing categorical with mode
    for col in df.select_dtypes(include='object'):
        df[col].fillna(df[col].mode()[0], inplace=True)
    # Fill missing numeric with median
    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col].fillna(df[col].median(), inplace=True)
    return df

def convert_dob_to_age(df):
    # Assuming date_of_birth is in 'dd-mm-yyyy' format
    from datetime import datetime
    today = datetime.today()
    def calc_age(dob):
        try:
            dob_dt = datetime.strptime(dob, '%d-%m-%Y')
            return today.year - dob_dt.year - ((today.month, today.day) < (dob_dt.month, dob_dt.day))
        except:
            return np.nan
    df['age'] = df['date_of_birth'].apply(calc_age)
    df['age'].fillna(df['age'].median(), inplace=True)
    df.drop('date_of_birth', axis=1, inplace=True)
    return df

def encode_features(df):
    categorical_cols = ['client_type', 'region', 'acquisition_purpose', 'referral_channel', 'country']
    df = pd.get_dummies(df, columns=categorical_cols)
    return df

def scale_features(df):
    scaler = StandardScaler()
    numeric_cols = ['floor_area_sqft', 'sale_price', 'age', 'satisfaction_score']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def preprocess(df):
    df = handle_missing(df)
    df = convert_dob_to_age(df)
    df = encode_features(df)
    df = scale_features(df)
    return df
