import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

import joblib

df = pd.read_csv("Social_Network_Ads.csv")

x = df.drop(columns =['User ID', 'Gender', 'Purchased'])
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

model = RandomForestClassifier(n_estimators = 100, random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('Accuracy of Challenger model:', accuracy_score(y_test,y_pred))

joblib.dump(model, 'challenger.pkl')
joblib.dump(scaler, 'challenger_scaler.pkl')

print("\nChallenger model saved successfully.")