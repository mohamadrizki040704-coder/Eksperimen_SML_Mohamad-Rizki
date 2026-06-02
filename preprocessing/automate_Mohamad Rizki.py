import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("PROSES PREPROCESSING DIMULAI")
print("")


base_path = "D:/Eksperimen_SML_MOHAMAD RIZKI"

df = pd.read_csv(f"{base_path}/titanic_raw/train.csv")
print(f"Data loaded: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("")

df.rename(columns={
    'Survived': 'survived',
    'Pclass': 'pclass',
    'Sex': 'sex',
    'Age': 'age',
    'SibSp': 'sibsp',
    'Parch': 'parch',
    'Fare': 'fare',
    'Embarked': 'embarked',
    'Name': 'name',
    'Cabin': 'cabin'
}, inplace=True)

df['embarked'] = df['embarked'].fillna('S')
df['age'] = df['age'].fillna(df['age'].median())


df['FamilySize'] = df['sibsp'] + df['parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)


df['sex'] = (df['sex'] == 'male').astype(int)
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'embarked_Q', 'embarked_S']
X = df[feature_cols]
y = df['survived']


scaler = StandardScaler()
scale_cols = ['age', 'fare', 'sibsp', 'parch', 'FamilySize']
X[scale_cols] = scaler.fit_transform(X[scale_cols])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

output_dir = f"{base_path}/preprocessing/titanic_preprocessed"
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

with open(f"{output_dir}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("PREPROCESSING SELESAI")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"File tersimpan di {output_dir}")