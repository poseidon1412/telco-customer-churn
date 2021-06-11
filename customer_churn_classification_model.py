import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('datasets_13996_18858_WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.drop('customerID',axis=1,inplace=True)
df.loc[(df['TotalCharges'] == ' '),'TotalCharges'] = 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.loc[(df['Churn'] == 'No'),'Churn'] = 0
df.loc[(df['Churn'] == 'Yes'),'Churn'] = 1
df['Churn'] = pd.to_numeric(df['Churn'])
df.replace('','_',regex=True,inplace=True)
df_encoded = pd.get_dummies(df,columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], drop_first=True)

X = df_encoded[['tenure', 'MonthlyCharges','PaperlessBilling_Yes','PaymentMethod_Electronic check']]
y = df_encoded["Churn"]
X.columns = X.columns.str.replace(' ','_')

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42,stratify=y)
#Feature scaling to scale the data or commonly known as normalization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = xgb.XGBClassifier(objective='binary:logistic',
                            seed = 42,
                            gamma=0.25,
                            learn_rate=0.1,
                            max_depth = 4,
                            reg_lambda=10,
                            scale_pos_weight=3,
                            subsample=0.9,
                            colsample_bytree=0.5)
model.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric= 'aucpr',
            eval_set=[(X_test,y_test)])

pickle.dump(model,open('Customer_churn_classifier.pkl','wb'))