import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

#load the data
world=pd.read_csv('world.csv',encoding = 'ISO-8859-1')
life=pd.read_csv('life.csv',encoding = 'ISO-8859-1')
df_world = pd.DataFrame(world)
df_life = pd.DataFrame(life)

#record the test result
test_result = []

#link the data sets, remove data with missing value, sorting
df_merged = pd.merge(left=df_life, right=df_world, how='left', left_on=['Country Code'], right_on=['Country Code'])
df_merged = df_merged.drop(['Country Name', 'Time'], axis = 1)
df = df_merged.sort_values('Country')

#get the class label
class_label = df['Life expectancy at birth (years)']

#replace .. with nan 
df = df.replace(r'\.{2}', np.nan, regex=True)
#get only the features
data = df.drop(['Country', 'Country Code', 'Year', 'Life expectancy at birth (years)'], axis = 1).astype(float)

#randomly select 70% of the instances to be training and the rest to be testing
X_train, X_test, y_train, y_test = train_test_split(data, class_label, train_size=0.7, test_size=0.3, random_state=200)


#perform median imputation to impute missing values
imp_median = SimpleImputer(missing_values=np.nan, strategy='median').fit(X_train)
X_train = imp_median.transform(X_train)
medians = imp_median.statistics_

#Scale each feature by removing the mean and scaling to unit variance
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
means = scaler.mean_
variances = scaler.var_

#perform median inputation to missing data on test set
X_test = imp_median.transform(X_test)

#Scale each feature by removing the mean and scaling to unit variance
X_test = scaler.transform(X_test)

index = 0
for feature in data:
    record = []
    record.append(feature)
    record.append(round(medians[index], 3))
    record.append(round(means[index], 3))
    record.append(round(variances[index], 3))
    test_result.append(record)
    index = index +1

## K-NN with k=3
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_score_knn3 = accuracy_score(y_test, y_pred)

## K-NN with k=7
knn = neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_score_knn7 = accuracy_score(y_test, y_pred)

## decision tree with max depth of 3
dt = DecisionTreeClassifier(criterion="entropy", random_state=200, max_depth=3)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
acc_score_dt3 = accuracy_score(y_test, y_pred)

print("Accuracy of decision tree: %5.3f" %(acc_score_dt3))
print("Accuracy of k-nn (k=3): %5.3f" %(acc_score_knn3))
print("Accuracy of k-nn (k=7): %5.3f" %(acc_score_knn7))

column_names = ["feature", "median", "mean", "variance"]
result = pd.DataFrame(test_result, columns = column_names)
result.to_csv('task2a.csv', index= False)