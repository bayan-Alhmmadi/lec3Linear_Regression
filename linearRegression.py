import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn

df= pd.read_csv('./dataSets/StudentsPerformance.csv')

df
print(df.isnull().sum())
df.info()
# Rename columns to remove spaces and special characters
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('/', '_')

print("\nRenamed columns:")
print(df.columns)
# Identify categorical columns
categorical_cols = [
    'gender',
    'race_ethnicity',
    'parental_level_of_education',
    'lunch',
    'test_preparation_course'
]

# Apply one-hot encoding
df_cleaned = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Display the cleaned DataFrame
print("\nCleaned and encoded DataFrame head:")
print(df_cleaned.head())
print("\nData types of the new DataFrame:")
df_cleaned.info()
df_cleaned.to_csv('StudentsPerformance_cleaned.csv', index=False)
#print("تم حفظ البيانات النظيفة في ملف StudentsPerformance_cleaned.csv بنجاح!")
df_cleaned.describe()
# (Target)
y = df_cleaned["math_score"]

# (Features)
X = df_cleaned[["reading_score", "writing_score"]]

print("شكل X:", X.shape)
print("شكل y:", y.shape)

import matplotlib.pyplot as plt 
#import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned["reading_score"], df_cleaned["math_score"], alpha=0.5, color="blue")

plt.title("Reading_Score vs Math_Score")
plt.xlabel("Reading_Score")
plt.ylabel("Math_Score")

plt.xlim(0, 100)   # لأن الدرجات بين 0 و 100
plt.ylim(0, 100)

plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df["writing_score"], df["math_score"], alpha=0.5, color="green")

plt.title("Writing_Score vs Math_Score")
plt.xlabel("Writing_Score")
plt.ylabel("Math_Score")

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.show()

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X, y)
regression.coef_ 
regression.intercept_
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned["reading_score"], df_cleaned["math_score"], alpha=0.5, color="blue")

plt.plot(X, regression.predict(X), color='red', linewidth=4)  # Add regression line

plt.title('Reading_Score vs Math_Score')
plt.xlabel('Reading_Score')
plt.ylabel(' Math_Score')
plt.ylim(0, 3000000000)  # Set y-axis limit for better visibility
plt.xlim(0, 450000000)  # Set x-axis limit for better
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned["writing_score"], df_cleaned["math_score"], alpha=0.5, color="blue")

plt.plot(X, regression.predict(X), color='red', linewidth=4)  # Add regression line

plt.title('Writing_score vs Math_Score')
plt.xlabel('Writing_score')
plt.ylabel(' Math_Score')
plt.ylim(0, 3000000000)  # Set y-axis limit for better visibility
plt.xlim(0, 450000000)  # Set x-axis limit for better
plt.show()
print(len(X), len(y))
regression.score(X, y)
#X has 1 features, but LinearRegression is expecting 2 features as input.
regression.predict([[50000000]])
