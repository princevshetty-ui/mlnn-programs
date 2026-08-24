import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

database = pd.read_csv('1stprogram_diabetes_dataset.csv')

df = database

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_with_zeros:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

print("Summary Statistics:")
print(df.describe())

print(f"\nMissing values:\n{df.isnull().sum()}")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

feature_cols = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

df[feature_cols] = scaler.fit_transform(df[feature_cols])

print("\nAfter normalization:\n", df.describe())

plt.figure(figsize=(15, 10))

plt.subplot(221)
df['Outcome'].value_counts().plot(kind='bar')
plt.title('Class Distribution')

plt.subplot(222)
sns.histplot(df['Glucose'], bins=30)
plt.title('Glucose Distribution')

plt.subplot(223)
sns.boxplot(x='Outcome', y='Age', data=df)
plt.title('Age vs Outcome')

plt.subplot(224)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()