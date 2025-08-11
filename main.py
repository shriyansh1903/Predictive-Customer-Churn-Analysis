import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(layout="wide")


@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        df['ChurnValue'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        return df
    except FileNotFoundError:
        st.error(f"Error: '{filepath}' not found. Please ensure the dataset is in the correct directory.")
        return None


@st.cache_resource
def train_model(df):
    df_model = df.drop(['customerID', 'Churn'], axis=1)
    X = df_model.drop('ChurnValue', axis=1)
    y = df_model['ChurnValue']
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model_pipeline.fit(X_train, y_train)
    return model_pipeline, X_test, y_test


st.title("📈 Telco Customer Churn Prediction Dashboard")
st.markdown(
    "An interactive dashboard to analyze customer churn, evaluate model performance, and identify actionable insights.")

df = load_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')

if df is not None:
    pipeline, X_test, y_test = train_model(df)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    st.sidebar.header("Dashboard Controls")
    show_data = st.sidebar.checkbox("Show Raw Data", False)
    show_model_performance = st.sidebar.checkbox("Show Model Performance Metrics", False)

    st.header("Business KPI Dashboard")

    results_df = X_test.copy()
    results_df['Actual_Churn'] = y_test
    results_df['Predicted_Churn'] = y_pred
    results_df['Churn_Probability'] = y_pred_proba

    predicted_churners_df = results_df[results_df['Predicted_Churn'] == 1]
    monthly_revenue_at_risk = predicted_churners_df['MonthlyCharges'].sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Customers Predicted to Churn", value=len(predicted_churners_df))
    with col2:
        st.metric(label="Monthly Revenue at Risk", value=f"${monthly_revenue_at_risk:,.2f}")
    with col3:
        st.metric(label="Model Accuracy", value=f"{pipeline.score(X_test, y_test):.2%}")

    st.markdown("---")

    col1, col2 = st.columns((1, 1))

    with col1:
        st.header("Visual Insights")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='ChurnValue', y='tenure', data=df, ax=ax1)
        ax1.set_title('Customer Tenure vs. Churn Status')
        ax1.set_xlabel('Churn (1 = Yes, 0 = No)')
        ax1.set_ylabel('Tenure (Months)')
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Contract', hue='ChurnValue', data=df, ax=ax2)
        ax2.set_title('Churn Count by Contract Type')
        ax2.set_xlabel('Contract Type')
        ax2.set_ylabel('Number of Customers')
        st.pyplot(fig2)

    with col2:
        st.header("Actionable Customer Insights")
        st.markdown(
            "Below are the customers with the highest probability of churning. These are prime candidates for a targeted retention campaign.")

        high_risk_customers = results_df.sort_values(by='Churn_Probability', ascending=False).head(15)
        st.dataframe(high_risk_customers[['tenure', 'Contract', 'MonthlyCharges', 'Churn_Probability']], height=950)

    if show_data:
        st.markdown("---")
        st.header("Raw Customer Data")
        st.dataframe(df)

    if show_model_performance:
        st.markdown("---")
        st.header("Detailed Model Performance")

        col1, col2 = st.columns(2)
        with col1:
            st.text("Confusion Matrix")
            st.text(confusion_matrix(y_test, y_pred))
        with col2:
            st.text("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
