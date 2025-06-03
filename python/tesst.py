
# import required libraries

import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

# read data from csv file
#read_file = pd.read_csv (r'C:/Users/ENGR. B.K. NUHU/Desktop/IMPLEMENTATIONS/DC_algorithm/entropies.txt')
#read_file.to_csv (r'C:/Users/ENGR. B.K. NUHU/Desktop/IMPLEMENTATIONS/DC_algorithm/entropies.csv', index=None)
#model_in = open('C:/Users/ENGR. B.K. NUHU/Desktop/IMPLEMENTATIONS/sdwsn-new-arima/DT_model.pickle','rb')
#model = pickle.load(model_in)
'''with open('svm_.txt', newline='') as txt_file:
    csv_file = csv.reader(txt_file, delimiter=' ')
    #if you need to convert it to csv file:
    with open ('DC_data.csv', "w",newline='') as new_csv_file:
        new_csv = csv.writer(new_csv_file, delimiter=',')
        for row in csv_file:
            print(' '.join(row))
            new_csv.writerow(row)'''

data = pd.read_csv('DC_data.csv')
data = data.drop_duplicates(subset= ['entropy', 'deviation', 'MR'])
#with open('clean_data.txt', 'a') as f:
                        #f.write(str(data) +'\n')

#print(data)

# splitting data into training and test set
training_set,test_set = train_test_split(data,test_size=0.2, random_state=20)
#print("train:",training_set)
#print("test:",test_set)

# prepare data for applying it to svm
x_train = training_set.iloc[:,0:2].values  # data
y_train = training_set.iloc[:,2].values  # target
x_test = test_set.iloc[:,0:2].values  # data
y_test = test_set.iloc[:,2].values  # target 
#print(x_train,y_train)
#print(x_test,y_test)
# fitting the data (train a model)
# Perform random sampling
rus = RandomOverSampler(random_state=20)
x_train_rus, y_train_rus = rus.fit_resample(x_train, y_train)

#plt.figure(figsize=(8, 6))
#plt.plot(y_train_rus)
#plt.title('Classes distribution')
#plt.legend(title='MR', loc='upper right')
# Show the plots
#plt.show()

classifier = SVC(kernel='rbf',random_state=1,C=1,gamma='auto')
classifier.fit(x_train_rus,y_train_rus)
clf = LogisticRegression();
clf.fit(x_train,y_train)
clf2 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=10, min_samples_leaf=3,
            min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='random');
clf2.fit(x_train_rus,y_train_rus)
clf3 = RandomForestClassifier(criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=10, min_samples_leaf=3,
            min_samples_split=3);
clf3.fit(x_train_rus, y_train_rus)

#filename = 'RF_model.pickle'
#pickle.dump(clf3, open(filename, 'wb'))

# perform prediction on x_test data
#entropy = -13.861606014361987 
#deviation = 1.7507735578188072e-06
#y_pred = model.predict([[entropy, deviation]])
#test_set['prediction']=y_pred
#print(y_pred)
##false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_pred)
# creating confusion matrix and accuracy calculation
#cm = confusion_matrix(y_test,y_pred)
#print(cm)
#print(classification_report(y_test,y_pred))
##accuracy = float(cm.diagonal().sum())/len(y_test)
#print('model accuracy is:',accuracy*100,'%')
Y_pred1 = classifier.predict(x_test)   #SVM
Y_pred2 = clf.predict(x_test)     #LR
Y_pred3 = clf2.predict(x_test)   #DT
Y_pred4 = clf3.predict(x_test)   #RF
#print(Y_pred2)
print(classification_report(y_test,Y_pred1))
cm1 = confusion_matrix(y_test,Y_pred1)
TN = cm1[0][0]
FN = cm1[1][0]
TP = cm1[1][1]
FP = cm1[0][1]
print('FP is ', FP)
print('FN is ', FN)
print('TP is ', TP)
print('TN is ', TN)
print(cm1)
accuracy1 = accuracy_score(y_test, Y_pred1)
print('model accuracy1 is:',accuracy1*100,'%')

print(classification_report(y_test,Y_pred2))
cm2 = confusion_matrix(y_test,Y_pred2)
TN = cm2[0][0]
FN = cm2[1][0]
TP = cm2[1][1]
FP = cm2[0][1]
print('FP is ', FP)
print('FN is ', FN)
print('TP is ', TP)
print('TN is ', TN)
print(cm2)
accuracy2 = accuracy_score(y_test, Y_pred2)
print('model accuracy2 is:',accuracy2*100,'%')

print(classification_report(y_test,Y_pred3))
cm3 = confusion_matrix(y_test,Y_pred3)
TN = cm3[0][0]
FN = cm3[1][0]
TP = cm3[1][1]
FP = cm3[0][1]
print('FP is ', FP)
print('FN is ', FN)
print('TP is ', TP)
print('TN is ', TN)
print(cm3)
accuracy3 = accuracy_score(y_test, Y_pred3)
print('model accuracy3 is:',accuracy3*100,'%')

print(classification_report(y_test,Y_pred4))
cm4 = confusion_matrix(y_test,Y_pred4)
TN = cm4[0][0]
FN = cm4[1][0]
TP = cm4[1][1]
FP = cm4[0][1]
print('FP is ', FP)
print('FN is ', FN)
print('TP is ', TP)
print('TN is ', TN)
print(cm4)
accuracy4 = accuracy_score(y_test, Y_pred4)
print('model accuracy4 is:',accuracy4*100,'%')
#set up plotting area
plt.figure(0).clf()

#fit logistic regression model and plot ROC curve
#model = LogisticRegression()
#model.fit(X_train, y_train)
#y_pred = model.predict_proba(X_test)[:, 1]
fpr1, tpr1, threshold1 = roc_curve(y_test, Y_pred1)
#fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, Y_pred1), 4)
plt.plot(fpr1,tpr1,label="SVM, AUROC="+str(auc))

#fit gradient boosted model and plot ROC curve
#model = GradientBoostingClassifier()
#model.fit(X_train, y_train)
#y_pred = model.predict_proba(X_test)[:, 1]
fpr2, tpr2, threshold2 = roc_curve(y_test, Y_pred2)
#fpr, tpr, _ = metrics.roc_curve(y_test, Y_pred2)
auc = round(metrics.roc_auc_score(y_test, Y_pred2), 4)
plt.plot(fpr2,tpr2,label="LR, AUROC="+str(auc))

fpr3, tpr3, threshold2 = roc_curve(y_test, Y_pred3)
#fpr, tpr, _ = metrics.roc_curve(y_test, Y_pred2)
auc = round(metrics.roc_auc_score(y_test, Y_pred3), 4)
plt.plot(fpr3,tpr3,label="DT, AUROC="+str(auc))

fpr4, tpr4, threshold2 = roc_curve(y_test, Y_pred4)
#fpr, tpr, _ = metrics.roc_curve(y_test, Y_pred2)
auc = round(metrics.roc_auc_score(y_test, Y_pred4), 4)
plt.plot(fpr4,tpr4,label="RF, AUROC="+str(auc))

#add legend
plt.legend()
plt.show()
