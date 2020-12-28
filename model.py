#Libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
import pickle

#dataset
loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )

loan_data.drop(columns=['Unnamed: 0',	'Loan_ID'], inplace=True)

#data with null instances
loan_data_null = loan_data[loan_data.isnull().any(axis=1)]

#Original Data without null
loan_data1 = loan_data.dropna()
loan_data1.reset_index(inplace=True)

ohe1 = OneHotEncoder(handle_unknown='ignore')
loan_data1_ohe = pd.DataFrame(ohe1.fit_transform(loan_data1[['Gender', 'Married',	'Dependents',	'Education','Self_Employed', 'Property_Area']]).toarray())

#match with ohe2
origin = [3, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18]
fill = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

ohe_conditioned = loan_data1_ohe.copy()

for idx in origin:
  for num in fill:
    ohe_conditioned[idx] = loan_data1_ohe[num]

for idx in [2, 4, 10, 14]:
  ohe_conditioned[idx] = 0

loan_data1 = loan_data1.drop(columns = ['index','Gender', 'Married',	'Dependents',	'Education','Self_Employed', 'Property_Area'], axis=1)

loan_data1 = loan_data1.join(ohe_conditioned)

#allocate test data
X_data1 = loan_data1.drop(columns='Loan_Status', axis=1)
y_data1 = loan_data1.Loan_Status

X_train, X_test, y_train, y_test = train_test_split(X_data1, y_data1, test_size=0.1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#deal with null data (Gender, Married, Dependent, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History)
null_categorical = ['Gender', 'Married', 'Dependent', 'Self_Employed']
null_numerical = ['LoanAmount',	'Loan_Amount_Term',	'Credit_History']

for column in null_numerical:
  loan_data_null[column] = loan_data_null[column].fillna(loan_data_null[column].mean())

loan_data_null.fillna('Not Specified', inplace=True)
loan_data_null.reset_index(inplace=True)

#ohe null original data
ohe2 = OneHotEncoder(handle_unknown='ignore')
X_ohe = pd.DataFrame(ohe2.fit_transform(loan_data_null[['Gender', 'Married',	'Dependents',	'Education','Self_Employed', 'Property_Area']]).toarray())

data_df = loan_data_null.drop(columns = ['Gender', 'Married',	'Dependents',	'Education','Self_Employed', 'Property_Area'], axis=1)
concat_data = data_df.join(X_ohe)

X_concat = concat_data.drop(columns=['Loan_Status','index'])
y_concat = concat_data.Loan_Status
X = X_train.append(X_concat)
y = y_train.append(y_concat)
X = pd.DataFrame(X)

#SMOTE
sm = SMOTE(random_state=30)
X_train, y_train = sm.fit_sample(X, y.ravel())

#Feature selection
drop_feature = [1,3,5,9,15]
X_train_new = pd.DataFrame(X_train)
X_train_new.drop(X_train_new.columns[drop_feature], axis = 1, inplace = True)
X_test_new = X_test.copy()
X_test_new.drop(X_test_new.columns[drop_feature], axis = 1, inplace = True)


#Model Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train_new, y_train)
y_predict = gnb.predict(X_test_new)
accuracy = sklearn.metrics.accuracy_score(y_test,y_predict)
f1 = sklearn.metrics.f1_score(y_test, y_predict)

print('Naive Bayes Accuracy: ', accuracy*100, '%')
print('Naive Bayes F1: ', f1*100, '%')

#Deployment
model_file = "naive_bayes.pkl"
with open(model_file, 'wb') as file:
  pickle.dump(gnb, file)

#load model back from file
with open(model_file, 'rb') as file:
  gnb_model = pickle.load(file)

# print(gnb_model)