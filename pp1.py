import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("loan_approval_dataset.csv", skipinitialspace = True)
#print(df.head())
df["loan_status"] = np.where(df["loan_status"] == "Approved", 1, 0)
df["education"] = np.where(df["education"] == "Graduate", 1, 0)
df["self_employed"] = np.where(df["self_employed"] == "Yes", 1, 0)
#print(df.head())
id = df["loan_id"]
y = df["loan_status"]
X = df.drop(["loan_status", "loan_id"], axis="columns")

X_train, X_test, y_train, y_test = train_test_split(X,y,
                          random_state=104,
                          train_size=0.75, shuffle=True)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Model score: {model.score(X_test, y_test)}")
print(metrics.confusion_matrix(y_test, pred))
