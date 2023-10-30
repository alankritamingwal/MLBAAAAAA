#importing modules
# from sklearn import svm
import sys

import numpy as np
import pandas
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

"""##Command Line arguments"""
print("enter training data file name")
train=input()
print("enter testing data file name")
test=input()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

with open(train, 'r') as temp_f:
    # COUNT THE NO OF COLUMNS IN EACH LINE
    col_count = [ len(l.split(",")) for l in temp_f.readlines() ]
  

# Generate the columns 0 to max-1 
column_names = [i for i in range(0, max(col_count))]

#read training data 
df = pd.read_csv(train, header=None, delimiter=",", names=column_names)

df = df.replace(np.nan, 0)
df = df[1:]

with open(test, 'r') as temp_f:
    # get No of columns in each line
    col_count = [ len(l.split(",")) for l in temp_f.readlines() ]


column_names = [i for i in range(0, max(col_count))]

### Read test csv
dft = pd.read_csv(test, header=None, delimiter=",", names=column_names)

dft = dft.replace(np.nan, 0)
dft= dft[1:].values


"""##Splitting Training dataset"""

#splitting the dataset into xtrain and ytrain
y_train = df.values[:,0].astype('int')
x_train = df.values[:,1:].astype('float')

np.unique(y_train, return_counts=True)

x_test = dft[:, 1:].astype('float')
x_test.shape


from sklearn.linear_model import LogisticRegression

# from sklearn.ensemble import GradientBoostingClassifier

#We are using Stacking Classifier
for rs in [10, 20, 30]:
  # models to be used at Level0 of stack classifier
  level0 = []                     #Defining levels of classifier
  mi = 10000
  level0.append(('lr', LogisticRegression(max_iter=mi)))
  level0.append(('knn', KNeighborsClassifier()))
  level0.append(('rf', RandomForestClassifier()))
  level0.append(('svm', SVC(max_iter=mi)))
  level0.append(('bayes', GaussianNB()))
  # defining the last level of stack classifier which is a meta learner model
  level1 = LogisticRegression(max_iter=mi)
  # define the Level1 of stacking
  model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

  # Training ML model 
  model.fit(x_train, y_train)

  # Predicting on the testing dataset
  y_test_pred = model.predict_proba(x_test)
  # taking the probo of getting 1
  y_test_pred = y_test_pred[:,1]

  # final output
  y_final = ['Labels']

  for i in range(0, len(y_test_pred)):
    y_final.append(y_test_pred[i])

  #Storing final output
  df2 = pd.read_csv("kaggle_test.csv", header = None).loc[:, :2]
  df2[2] = y_final[:]
  df2 = df2.drop(1, axis = 1)
  df2 = df2.values

  #Saving the output
  pd.DataFrame(df2).to_csv("result_"+str(rs)+".csv", header = None, index = None)
  print("File Created - result_", rs)

print("\nFinished!!")
print("NOTE: result_30 is our final file.")

#Trial_10 gives 81.997% accuracy on kaggle.
#Trial_20 gives 83.335% accuracy on kaggle.
#Trail_30 gives 84.183% accuracy on kaggle which is our highest accuracy.

"""###Other models that we have tried"""

""" TRIAL-1
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(bootstrap= False, max_depth= 20, max_features= 'auto', min_samples_leaf= 1, min_samples_split= 10, random_state= 1)
# clf = RandomForestClassifier(bootstrap= False, max_depth= 7, max_features='auto', min_samples_leaf= 1, min_samples_split= 5, random_state= 2)

clf.fit(X_train_chi,y_train)
predictions=clf.predict(X_test_chi)
# df_merged = pd.merge(data_test_ID, pd.DataFrame(predictions ))
# df_merged['ID'] =data_test_ID
# data_test_ID=data_test_IDpredictions
# print(data_test_ID.head())
print(predictions.shape,data_test_ID.shape)
prediction = pd.DataFrame({'Id':data_test_ID,'Label': predictions})
# print(prediction)
# pd.to_csv('check.csv')
prediction.to_csv (r'output1.csv', index = False, header=True)

NOTE: It is giving a accuracy of 72%. 
"""

""" TRIAL-2
from sklearn.ensemble import ExtraTreesClassifier

for rs in [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]:
# TOP_FEATURES = 15

  forest = ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=rs)
  forest.fit(x_train, y_train)

  importances = forest.feature_importances_
  std = np.std(
      [tree.feature_importances_ for tree in forest.estimators_],
      axis=0
  )
  indices = np.argsort(importances)[::-1]
  indices = indices[:19]
  print(len(indices))
  print(len(importances))
  print('Top features:')
  list = []
  for f in range(19):
    list.append(indices[f])
    print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))

  x_train_2 = x_train[:,list]
  x_test_2 = x_test[:,list]

  models = [
      LogisticRegression(),
      XGBClassifier(max_depth=2),
      GradientBoostingClassifier(learning_rate=0.001, n_estimators=300),
      RandomForestClassifier(max_depth=2, max_features='log2', n_estimators=50),

  ]

  preds = pd.DataFrame()
  for i, m in enumerate(models):
      m.fit(x_train_2, y_train),
      preds[i] = m.predict_proba(x_test_2)[:,1]

  # weights = [1, 0.8, 0.6, 0.4]
  weights = [1, 0.8, 0.8, 0.6]
  # weights = [1, 0.8, 0.8, 0.8]
  # weights = [1, 0.8, 0.4, 0.6]
  y_test_pred= (preds * weights).sum(axis=1) / sum(weights)
  print(y_test_pred)

  # final array for output
  y_final = ['Labels']

  for i in range(0, len(y_test_pred)):
    y_final.append((y_test_pred[i]))

  # dataframe to store the final output
  df2 = pd.read_csv("kaggle_test.csv", header = None).loc[:, :2]
  df2[2] = y_final[:]
  df2 = df2.drop(1, axis = 1)
  df2 = df2.values

  #Saving the output file
  pd.DataFrame(df2).to_csv("VA_R"+str(rs)+".csv", header = None, index = None)
  
  
NOTE: It is giving a accuracy of 76.378%. """


"""Trial-3  SVC
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.1, 1]
    gammas = [0.01, 0.1, 1]
    kernels=['linear','rbf']
    param_grid = {'C': Cs, 'gamma' : gammas,'kernel':kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds,scoring = 'roc_auc',verbose=3)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_,grid_search
params,clf=svc_param_selection(X_train_chi, y_train, 5)
print(params)
print("Cross Validation AUC Score:",clf.best_score_)

predictions=clf.predict(X_test_chi)

prediction = pd.DataFrame({'ID':data_test_ID,'Labels': predictions})

prediction.to_csv (r'output2.csv', index = False, header=True)
NOTE: It is giving a accuracy of 50%.

"""



