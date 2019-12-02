# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:14:31 2019

@author: Rakin Shahriar
"""
#import nltk
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.neural_network import MLPClassifier
#import matplotlib.pyplot as plt
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import json
from sklearn.externals import joblib 
os.chdir(r'C:\Users\Rakin Shahriar\Desktop\soup\article');
data = json.load(open('data.json',encoding = 'utf-8'))
#data = pd.read_json('data.json',orient='table')
#x = data['token'].values.tolist()
#y = data['class'].values.tolist()
print(type(data['data-set']['record'][0]['token']))
data = data['data-set']['record']
df = pd.DataFrame(data)
print(df,df.shape)
print(type(df))
# e porjonto ampan data python e feed kora  holo.
#print(df['token'])
token_exp = np.array(df['token'])
# evabe numpy array te nilam 
# ebar run korlam
print(token_exp[19])
print(type(token_exp))
i = 1
t = len(token_exp)
s = t-1
#listit = []
"""
while i <= t:
    if (i+1) < t and (i-1)>=1 and (i+2) < t and (i-2)>=1:
       print(token_exp[i-2],token_exp[i-1],sep='_')
       print(token_exp[i-1],token_exp[i],token_exp[i+1],sep=' ')
       print(token_exp[i+2],token_exp[i+1],sep='_')
       df['prev'] = token_exp[i-1]
       df['token'] = token_exp[i]
       df['next'] = token_exp[i+1]
    i = i+1
print(df.token)"""
#while i <= t:
df['next'] = pd.DataFrame(token_exp[1:t])
token_list = list(token_exp)
token_list.insert(0,'null')
token_list.pop()
df['previous'] = pd.DataFrame(token_list)
"""
item_list = []
for index,rows in df.iterrows():
    Item_list = [rows.token, rows.next, rows.previous]
    item_list.append(Item_list)
print(item_list)"""
fet_dic = df.loc[:,df.columns !='class'].to_dict()
print(fet_dic)
#fet_dic = {}
#for i in range(len(token_exp)):
#    fet_dic[token_exp[i]] = token_exp[i]
print(fet_dic)
print(type(fet_dic))
#print(type(df.prev))
#feature = vectorizer.get_feature_names()
diclist = []
itemtemp = []
for key,value in fet_dic.items():
    temp = [key,value]
    diclist.append(temp)
print(diclist)
#label = ['o','Pname','Tname','Lname','Aname','Oname']
label = list(df['class'])
vectorizer = LabelEncoder()
y = vectorizer.fit_transform(label)
#df['class'] = y_nw
#y = df['class']
#vector = OneHotEncoder()
#x = vector.fit_transform(diclist)
#print(x)
cf = pd.DataFrame(fet_dic)
x = pd.get_dummies(cf)
cetagorical = ['token','next','previous']
x = x.to_numpy()
#x = pd.get_dummies(df[columns=cetagorical])
#print(x)
"""
print(type(y))
print(df)
print("0th token = ",token_exp[0])
print("rest")
print(df.shape)
print(df['class'])
print("Type of dummy is : ",type(dummy))"""
print(type(y))
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.25,random_state=40)
classifier = MLPClassifier(activation="logistic",solver='sgd',alpha=0.1,hidden_layer_sizes=(5,15))
classification = classifier.fit(X_train,Y_train)
# Save the model as a pickle in a file 
joblib.dump(classifier, 'NeuralNet.pkl') 
  
# Load the model from the file 
nn_from_joblib = joblib.load('NeuralNet.pkl')
Y_pred = nn_from_joblib.predict(X_test)
confusion = confusion_matrix(Y_test,Y_pred)
print("confusion matrix : \n",confusion)
accuracy = accuracy_score(Y_test,Y_pred)*100
print("System accuracy = ",accuracy)
c = precision_score(Y_test, Y_pred, average='macro')*100
print("Precission of the system = ",c)
d = recall_score(Y_test,Y_pred,average='micro')*100
print("Recall of the system = ",d)
u = f1_score(Y_test,Y_pred,average='macro')*100
print("F1 score of the system = ",u)
