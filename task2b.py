import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from operator import itemgetter 



#load the data sets
world=pd.read_csv('world.csv',encoding = 'ISO-8859-1')
life=pd.read_csv('life.csv',encoding = 'ISO-8859-1')
df_world = pd.DataFrame(world)
df_life = pd.DataFrame(life)

#link the data sets, remove data with missing value, sorting
df_merged = pd.merge(left=df_life, right=df_world, how='left', left_on=['Country Code'], right_on=['Country Code'])
df_merged = df_merged.drop(['Country Name', 'Time'], axis = 1)
df = df_merged.sort_values('Country')

#replace .. with nan 
df = df.replace(r'\.{2}', np.nan, regex=True)

#get the class label
class_label = df['Life expectancy at birth (years)']
#get only the features
org_features = df.drop(['Country', 'Country Code', 'Year', 'Life expectancy at birth (years)'], axis = 1).astype(float)


#split the dataset into training ant testing
X_train, X_test, y_train, y_test = train_test_split(org_features, class_label, train_size=0.7, random_state = 200)


#perform imputation and scale to training test and test set separately
imp_median = SimpleImputer(missing_values=np.nan, strategy='median').fit(X_train)
i_X_train = pd.DataFrame(imp_median.transform(X_train))
i_X_train.columns = org_features.columns

i_X_test = pd.DataFrame(imp_median.transform(X_test))
i_X_test.columns = org_features.columns

#select first 4 features for later use
i_X_train_f4 = i_X_train.iloc[:, 0:4]
i_X_test_f4 = i_X_test.iloc[:, 0:4]

#interation term pairs algorithm
def interaction_term_pair(X):
    new_X_names = []
    new_X_datas = []
    
    for index, row in X.iterrows():
        new_X_data = []
        for i in range(len(X.columns)):
            for j in range(i+1, len(X.columns)):
                if (len(new_X_names) != 190):
                    X_name_a = X.columns[i]
                    X_name_b = X.columns[j]
                    new_X_name = X_name_a +" * "+X_name_b
                    new_X_names.append(new_X_name)
                one_X_data = row[i] * row[j]
                new_X_data.append(one_X_data)
        new_X_datas.append(new_X_data)
    X_interaction = pd.DataFrame(data=new_X_datas, columns=new_X_names)
    return X_interaction

def concat_addtional_feature(X1, X2, labels):
    X_211 = pd.concat([X1,X2], axis=1)
    X_211["cluster label"] = labels
    return X_211


# perform feature engineering: interaction term pairs
X_interaction_train = interaction_term_pair(i_X_train)
X_interaction_test = interaction_term_pair(i_X_test)

# perform feature engineering: k-means clustering on training set
#determine the optimal k value 
cost =[] 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, random_state=0).fit(i_X_train) 
    # calculates squared error for the clustered points
    cost.append(kmeans.inertia_)      
  
plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 
plt.title("The cost against K values")
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.savefig('task2bgraph1.png')
print("The cost vs k value graph shows that when optimal k value is 4\n")
#perform k means with k = 4
kmeans = KMeans(n_clusters=4, random_state=0).fit(i_X_train)
cluster_label_train = kmeans.labels_
cluster_label_test = kmeans.predict(i_X_test)
#combining 211 features
X_211_train = concat_addtional_feature(i_X_train,X_interaction_train, cluster_label_train)
X_211_test = concat_addtional_feature(i_X_test,X_interaction_test, cluster_label_test)

###perform 3NN on three different feature enginerring
knn = neighbors.KNeighborsClassifier(n_neighbors=3)

## select 4 features from 211 features
#normalize data
scaler = StandardScaler().fit(X_211_train)
n_X_211_train = pd.DataFrame(scaler.transform(X_211_train))
n_X_211_train.columns = X_211_train.columns
n_X_211_test = pd.DataFrame(scaler.transform(X_211_test))
n_X_211_test.columns = X_211_test.columns

# perform greedy approach on each single feature
single_feature_accs = {}
for i in range(len(n_X_211_train)):
    single_feature_train = n_X_211_train.iloc[:, i].values.reshape(-1, 1)
    single_feature_test = n_X_211_test.iloc[:, i].values.reshape(-1, 1)
    
    knn.fit(single_feature_train, y_train)
    y_pred = knn.predict(single_feature_test)
    acc_single = accuracy_score(y_test, y_pred)
    col_name = n_X_211_train.columns[i]
    single_feature_accs[col_name] = acc_single
# find the best 4 features
best_four_features = dict(sorted(single_feature_accs.items(), key = itemgetter(1), reverse = True)[:4])
print("Best four features gained from 'greedy wrapper approach' (with accuracy):")
for key, value in best_four_features.items():
    print("feature: %s\n%s\n" %(key, value))

features_4_train = n_X_211_train[[n for n in best_four_features.keys()]]
features_4_test = n_X_211_test[[n for n in best_four_features.keys()]]
        
knn.fit(features_4_train, y_train)
y_pred = knn.predict(features_4_test)
acc_fe = accuracy_score(y_test, y_pred)

#scatter plot for the four features
train_set_g4 = pd.concat([features_4_train,y_train], axis=1)
train_set_g4.columns = ['featureA','featureB', 'featureC', 'featureD','class']
sns_plot = sns.pairplot(train_set_g4, hue='class', height = 1.5)
sns_plot.fig.suptitle("Correlation analysis between selected 4 features")
sns_plot.savefig("task2bgraph2.png")

##PCA on oringinal 20 features
scaler = StandardScaler().fit(i_X_train)
i_X_train = scaler.transform(i_X_train)
i_X_test = scaler.transform(i_X_test)

pca = PCA(n_components=4)
i_X_train_pca = pca.fit_transform(i_X_train)
i_X_test_pca = pca.transform(i_X_test)
explained_variance = pca.explained_variance_ratio_
cum_sum = pca.explained_variance_ratio_.cumsum()
print("Explained variances caused by each of the first 4 principal components:\n%s" %(explained_variance))
for i in range(1, 5):
    print("Cumulative sum of first %d principle components: %5.3f" %(i, cum_sum[i-1]))
print()
knn.fit(i_X_train_pca, y_train)
y_pred = knn.predict(i_X_test_pca)
acc_pca = accuracy_score(y_test, y_pred)

#scatter plot for PCA features
i_X_train_pca = pd.DataFrame(i_X_train_pca)
train_set_pca = pd.concat([i_X_train_pca,y_train], axis=1)
train_set_pca.columns = ['component1','component2', 'component3', 'component4','class']
sns_plot = sns.pairplot(train_set_pca, hue='class', height = 1.5)
sns_plot.fig.suptitle("Correlation analysis between 4 components PCA")
sns_plot.savefig("task2bgraph3.png")


##select first 4 featurers of oringinal features
scaler = StandardScaler().fit(i_X_train_f4)
i_X_train_f4 = scaler.transform(i_X_train_f4)
i_X_test_f4 = scaler.transform(i_X_test_f4)

knn.fit(i_X_train_f4, y_train)
y_pred = knn.predict(i_X_test_f4)
acc_f4 = accuracy_score(y_test, y_pred)

#scatter plot for the first four features
i_X_train_f4 = pd.DataFrame(i_X_train_f4)
train_set_f4 = pd.concat([i_X_train_f4,y_train], axis=1)
train_set_f4.columns = ['feature1','feature2', 'feature3', 'feature4','class']
sns_plot = sns.pairplot(train_set_f4, hue='class', height = 1.5)
sns_plot.fig.suptitle("Correlation analysis between first 4 features")
sns_plot.savefig("task2bgraph4.png")

print("Accuracy of feature engineering: %5.3f" %(acc_fe))
print("Accuracy of PCA: %5.3f" %(acc_pca))
print("Accuracy of first four features: %5.3f" %(acc_f4))