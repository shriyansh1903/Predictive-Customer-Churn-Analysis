import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --- Data Simulation ---
data = {
    'CustomerID': range(1000, 1500),
    'SignupDate': [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(500)],
    'LastActivityDate': [datetime(2025, 6, 1) + timedelta(days=np.random.randint(0, 60)) for _ in range(500)],
    'SubscriptionPlan': np.random.choice(['Basic', 'Premium', 'Pro'], 500, p=[0.6, 0.3, 0.1]),
    'MonthlyFee': np.random.uniform(10, 100, 500),
    'SupportTicketsCount': np.random.randint(0, 10, 500),
    'Age': np.random.randint(18, 65, 500)
}
df = pd.DataFrame(data)

# Introduce churn-like behavior for a fraction of the data
churn_indices = df.sample(frac=0.2, random_state=42).index
df.loc[churn_indices, 'LastActivityDate'] = df.loc[churn_indices, 'LastActivityDate'] - timedelta(days=120)
df.loc[churn_indices, 'SupportTicketsCount'] += np.random.randint(1, 5, len(churn_indices))

# --- Data Cleaning, Feature Engineering & Visualization ---
print("\n--- Data Info ---")
df.info()

print("\n--- Engineering New Features ---")
df['SignupDate'] = pd.to_datetime(df['SignupDate'])
df['LastActivityDate'] = pd.to_datetime(df['LastActivityDate'])

current_date = datetime(2025, 8, 12)
df['tenure_days'] = (current_date - df['SignupDate']).dt.days
df['days_since_last_activity'] = (current_date - df['LastActivityDate']).dt.days
df['churn'] = (df['days_since_last_activity'] > 90).astype(int)

print("\nDataFrame with new features and target variable:")
print(df.head())

# Visualize the relationship between tenure and churn
plt.figure(figsize=(8, 5))
sns.boxplot(x='churn', y='tenure_days', data=df)
plt.title('Customer Tenure vs. Churn Status')
plt.xlabel('Churn (1 = Yes, 0 = No)')
plt.ylabel('Tenure (Days)')
plt.show()


# Visualize churn by subscription plan
plt.figure(figsize=(8, 5))
sns.countplot(x='SubscriptionPlan', hue='churn', data=df)
plt.title('Churn Count by Subscription Plan')
plt.xlabel('Subscription Plan')
plt.ylabel('Number of Customers')
plt.show()


# --- Model Building & Evaluation ---
features = ['SubscriptionPlan', 'MonthlyFee', 'SupportTicketsCount', 'Age', 'tenure_days', 'days_since_last_activity']
target = 'churn'
X = df[features]
y = df[target]

categorical_features = ['SubscriptionPlan']
numerical_features = ['MonthlyFee', 'SupportTicketsCount', 'Age', 'tenure_days', 'days_since_last_activity']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("\n--- Model Evaluation Results ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))