# Task 1: Working with Churn Data
# 1.1 Import path utilities and pandas for reading the dataset.
# %%
print(f"1. Importing path utilities and pandas for reading the dataset.")
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Announce that the preprocessing script has started.
print("Starting Churn Modeling Dataset Preprocessing Pipeline... \n")

# Locate the input and output CSV files in the project's data directory.
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
raw_file = DATA_DIR / 'Churn_Modeling.csv'
processed_file = DATA_DIR / 'processed_Churn_Modeling.csv'

# %%
# 1.2. Read the raw churn dataset and report its dimensions.
print(f"2. Reading raw data from: {raw_file}...")
df = pd.read_csv(raw_file)
print(f"Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print()

# %%
# 1.3. Inspect data types and identify categorical and numerical features.
print("3. Inspecting data types and identifying feature categories...")
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"   • Categorical features: {categorical_features}")
print(f"   • Numerical features: {numerical_features}")
print()

# %%
# 1.4. Explore data distribution and identify missing or inconsistent values.
print("4. Exploring data distribution and identifying missing/inconsistent values...")
print(f"   • Missing values per column:\n{df.isnull().sum()}")
print(f"   • Inconsistent values per column:\n{df.nunique()}")
print()

# %%
# 1.4a. Check for Categorical Inconsistencies (Typos & Formatting)
# Inspect the actual text categories
print("Geography values:", df['Geography'].unique())
print("Gender values:", df['Gender'].unique())
# %%
# 1.4b. Logical Inconsistencies (Contradictions)
# # Check for impossible combinations (e.g., Minors with massive salaries)
# Check for impossible combinations (e.g., Minors with massive salaries)
impossible_salary = df[(df['Age'] < 18) & (df['EstimatedSalary'] > 50000)]
print(f"Logical errors (Minors with high salaries): {len(impossible_salary)}")

# %%
# 1.4c. Check for Numerical Inconsistencies (Outliers)
# Identify outliers in numerical features using IQR or Z-score methods
print("Checking for numerical inconsistencies...")
# Check the boundaries of your numerical data
print(df[['Age', 'CreditScore', 'Tenure', 'Balance']].describe().loc[['min', 'max']])

# %%
# 1.5. Perform grouping and pivot analysis on key variables such as Geography and Gender.
print("Performing grouping and pivot analysis...")
grouped_data = df.groupby(['Geography', 'Gender']).size()
print(grouped_data)

# %%
# 1.6. Generate summary statistics to understand customer churn patterns.
print("Generating summary statistics...")
summary_stats = df.groupby('Exited').agg({
    'Age': 'mean',
    'CreditScore': 'mean',
    'Tenure': 'mean',
    'Balance': 'mean'
})
print(summary_stats)    

# %%
# 1.7. Prepare insights for model training and reporting.
print("Preparing insights for model training and reporting...")
# Core Insights Synthesis for Model Training & Reporting
print("\n--- Synthesized Insights for Reporting & Training ---")
overall_churn = df["Exited"].mean()
print(f"1. Overall Churn Rate: {overall_churn:.2%}")

# Highlight geographical anomalies
geo_churn = df.groupby("Geography")["Exited"].mean()
for geo, rate in geo_churn.items():
    print(f"   - Churn Rate in {geo}: {rate:.2%}")

# Highlight age disparities (often a massive driver in churn modeling)
young_churn = df[df["Age"] < 40]["Exited"].mean()
older_churn = df[df["Age"] >= 40]["Exited"].mean()
print(f"2. Age Impact: Churn rate for customers under 40 is {young_churn:.2%}, "
      f"compared to {older_churn:.2%} for customers 40 and older.")

# Check balance impact
zero_balance_churn = df[df["Balance"] == 0]["Exited"].mean()
has_balance_churn = df[df["Balance"] > 0]["Exited"].mean()
print(f"3. Balance Impact: Churn rate for zero-balance accounts is {zero_balance_churn:.2%}, "
      f"compared to {has_balance_churn:.2%} for active-balance accounts.")

print("\n--- Modeling Recommendations ---")
print("- Drop high-cardinality non-informative columns: 'RowNumber', 'CustomerId', 'Surname'")
print(f"- Encode categorical variables {categorical_features} using One-Hot Encoding.")
print(f"- Scale skewed numerical features (e.g., 'Balance', 'EstimatedSalary') using StandardScaler.")
print(f"- Address class imbalance: Only {overall_churn:.2%} of the dataset represents the churned class.")
# %%
# 1. Drop high-cardinality non-informative columns
# The raw header repeats 'CustomerId', so pandas reads the real ID column as 'CustomerId.1'
columns_to_drop = ['RowNumber', 'CustomerId', 'CustomerId.1', 'Surname']

# Drop the columns in place to update the 'df' DataFrame directly
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print("Remaining columns in DataFrame:")
print(df.columns.tolist())

# %%
# List of categorical columns left to encode
categorical_cols = ['Geography', 'Gender']

# One-hot encode with drop_first=True to stay safe for ALL model types 
# This drops 'Male' and one country, using them as baseline references
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Columns after One-Hot Encoding:")
print(df_encoded.columns.tolist())


# %%
# Save Processed Data
print(f"Saving processed dataset to: {processed_file}")
processed_file.parent.mkdir(parents=True, exist_ok=True)
df_encoded.to_csv(processed_file, index=False)

print(f"\nPreprocessing completed successfully!")
print(f"   • Final dataset shape: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")

# %%
