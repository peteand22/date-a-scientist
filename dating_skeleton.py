import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score, accuracy_score, recall_score, precision_score, f1_score
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
#Create your df here:
df = pd.read_csv("profiles.csv")

startTime = datetime.now()
#print(df.drinks.value_counts())
#print(df.drugs.value_counts())
#print(df.smokes.value_counts())

smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes": 4}
df["smokes_code"] = df.smokes.map(smokes_mapping)
drugs_mapping = {"never": 0, "sometimes": 1, "often": 1}
df["drugs_code"] = df.drugs.map(drugs_mapping)
drinks_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drinks_mapping)
sex_mapping = {"f": 0, "m": 1}
df["sex_code"] = df.sex.map(sex_mapping)

#Linear Regression
print("LINEAR REGRESSION-")
startTime = datetime.now()

#Single
#x = df[['drinks_code']] #Uncomment/Comment to run Single vs Multiple

#Multiple
x = df[['drinks_code', 'smokes_code']] #Uncomment/Comment to run Single vs Multiple

#Data setup
x = np.nan_to_num(x)
#x = preprocessing.scale(x)
x_test = x[:-40000]
x_train = x[-40000:]

y = df.drugs_code
y = np.nan_to_num(y)
#y = preprocessing.scale(x)
y_test = y[:-40000]
y_train = y[-40000:]

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
 
plt.plot(x_train, y_train, 'o')
 
y_predict = []
y_predict = regr.predict(x_test)
y_score = regr.score(x_test, y_test)

#Coefficients
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predict))
print('Variance: %.2f' % r2_score(y_test, y_predict))
 
#plot
plt.plot(x_test, y_predict)
plt.show()
 
predict = regr.predict([[4, 1]])
print(predict)
endTime = datetime.now()
print(endTime - startTime)


#Classifier Data Setup
x2 = df[['drinks_code', 'smokes_code']]
x2 = np.nan_to_num(x2)
#x2 = preprocessing.scale(x2)
x2_test = x2[:-40000]
x2_train = x2[-40000:]

y2 = df.drugs_code
y2 = np.nan_to_num(y2)
#y2 = preprocessing.scale(y2)
y2_test = y2[:-40000]
y2_train = y2[-40000:]

#Naive Bayes Classifier
print("NAIVE BAYES CLASSIFIER-")
test = [[4,2]]

classifier = MultinomialNB()
classifier.fit(x2_train, y2_train)
y_pred2 = classifier.predict(x2_test)

print("Accuracy:")
acc_score = accuracy_score(y2_test, y_pred2)
print(acc_score)
print("Recall:")
rec_score = recall_score(y2_test, y_pred2)
print(rec_score)
print("Precision:")
prec_score = precision_score(y2_test, y_pred2)
print(prec_score)
print("F1_Score:")
f1_score = f1_score(y2_test, y_pred2)
print(f1_score)
endTime = datetime.now()
print(endTime - startTime)
print("Prediction Probability")
print(classifier.predict_proba(test))

#KNearest
print("K-NEAREST NEIGHBORS-")
clsfr = KNeighborsClassifier(n_neighbors = 3)
clsfr.fit(x2_train, y2_train)
kn_predict = clsfr.predict(x2_test)

print("Accuracy:")
acc_score = accuracy_score(y2_test, kn_predict)
print(acc_score)
print("Recall:")
rec_score = recall_score(y2_test, kn_predict)
print(rec_score)
print("Precision:")
prec_score = precision_score(y2_test, kn_predict)
print(prec_score)
print("F1_Score:")
#f1_score = f1_score(y_test, kn_predict)
#print(f1_score)
print("Prediction Probability")
print(clsfr.predict_proba(test))
print("Time:")
endTime = datetime.now()
print(endTime - startTime)

#Scratch Code-
# =============================================================================
# print(df.drugs.value_counts())
# x = df['sex_code']
# x = np.nan_to_num(x)
# print(x)
# zero = 0
# one = 0
# two = 0
# three = 0
# four = 0
# five = 0
# for i in x:
#     if i == 0:
#         zero +=1
#     elif i==1:
#         one += 1
#     elif i==2:
#         two += 1
#     elif i==3:
#         three +=1
#     elif i==4:
#         four +=1
#     elif i==5:
#         five +=1
# print(zero, one, two, three, four, five)
# plt.hist(x)
# plt.xlabel("response value")
# plt.ylabel("frequency")
# plt.xlim(0, 1)
# plt.show()
# =============================================================================
#average_precision = average_precision_score(x, x)
#print("Average precision-recall score: %.2f" % average_precision)

#Question I abandoned

#Linear Regression Sex by Height
#Sex mapping
# =============================================================================
#education_mapping = {"graduated from college/university": 0, "graduated from masters program": 1, "working on college/university": 2, "working on masters program": 3, "graduated from two-year college": 4, "graduated from high school": 5, "graduated from ph.d program": 6, "graduated from law school": 7, "working on two-year college": 8, "dropped out of college/university": 9, "working on ph.d program": 10, "college/university": 11, "graduated from space camp": 12, "dropped out of space camp": 13, "graduated from med school": 14, "working on space camp": 15, "working on law school": 16, "two-year college": 17, "working on med school": 18, "dropped out of two-year college": 19, "dropped out of masters program": 20, "masters program": 21, "dropped out of ph.d program": 22, "dropped out of high school": 23, "high school": 24, "working on high school": 25, "space camp": 26, "ph.d program": 27, "law school": 28, "dropped out of law school": 29, "dropped out of med school": 30, "med school": 31}
#df["education_code"] = df.education.map(education_mapping)
# sex_mapping = {"m": 0, "f": 1}
# df["sex_code"] = df.sex.map(sex_mapping)
# #print(df.dtypes)
# x = df.height
# x = np.array(x).reshape((-1, 1))
# x = np.nan_to_num(x)
# x = preprocessing.scale(x)
# y = np.array(df.sex_code)
# y = np.nan_to_num(y)
# y = preprocessing.scale(y)
# #Create training and 
# x_test = x[:-40000]
# x_train = x[-40000:]
# y_test = y[:-40000]
# y_train = y[-40000:]
# 
# #print(x_test)
# #print(y_test)
# 
# regr = linear_model.LinearRegression()
# #Train
# regr.fit(x_train, y_train)
# 
# #Predict
# y_predict = regr.predict(x_test)
# 
# print('Coefficients: \n', regr.coef_)
# print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predict))
# print('Variance: %.2f' % r2_score(y_test, y_predict))
# 
# #plot
# plt.scatter(x_test, y_test,  color='blue')
# plt.plot(x_test, y_predict, color='red', linewidth=5)
# 
# plt.xticks(())
# plt.yticks(())
# 
# plt.show()
# 
# #Multiple Linear Regression Sex by Height, Income
# x2 = df[['height','income']]
# x2 = np.nan_to_num(x2)
# x2 = preprocessing.scale(x2)
# x2_test = x[:-40000]
# x2_train = x[-40000:]
# 
# regr2 = linear_model.LinearRegression()
# regr2.fit(x2_train, y_train)
# y2_predict = regr2.predict(x2_test)
# plt.scatter(x2_test, y_test,  color='blue')
# plt.plot(x2_test, y2_predict, color='red', linewidth=5)
# 
# plt.xticks(())
# plt.yticks(())
# plt.show()
# =============================================================================