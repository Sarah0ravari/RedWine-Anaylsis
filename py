import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.naive_bayes   import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm   import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble   import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline


import warnings
warnings.simplefilter("ignore")

# read from file
data = pd.read_csv("redwine.csv")

#drop white wine
data.drop(data.loc[data['type']=="white"].index, inplace=True)
data = data.drop('type', axis=1)

print(data.info());

#print unique values
print("Number of unique values in each column:\n")
for i in data.columns:
    print(i, len(data[i].unique()))
    
    
#create binary labled data    
data['bin_quality'] = pd.cut(data['quality'], bins=[0, 6.5, 10], labels=["bad", "good"])
    
data.head(10)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

data_length = len(data)
quality_percentage = [100 * i / data_length for i in data["quality"].value_counts()]
bin_quality_percentage = [100 * i / data_length for i in data["bin_quality"].value_counts()]

#heat map
sns.countplot("quality", data=data, ax=ax[0, 0])
sns.countplot("bin_quality", data=data, ax=ax[0, 1]);

sns.barplot(x=data["quality"].unique(), y=quality_percentage, ax=ax[1, 0])
ax[1, 0].set_xlabel("quality")

sns.barplot(x=data["bin_quality"].unique(), y=bin_quality_percentage, ax=ax[1, 1])
ax[1, 1].set_xlabel("bin_quality")


        
plt.figure(figsize=[15, 15])
sns.heatmap(data.corr(), xticklabels=data.columns[:-1], yticklabels=data.columns[:-1], square=True, cmap="Spectral_r", center=0);



#A set of classifiers

model_names = ['LogisticRegression',
               'KNeighborsClassifier',
               'SVC',
               'MLPClassifier',
               'ExtraTreesClassifier',
               'RandomForestClassifier',
               'LinearDiscriminantAnalysis']

classifiers = [LogisticRegression, 
               KNeighborsClassifier, 
               SVC,
               MLPClassifier, 
               ExtraTreesClassifier,#randomize
               RandomForestClassifier, 
                LinearDiscriminantAnalysis] 

def cross_val_mean_std(clsf, data, labels, cv=5):
    cross_val = cross_val_score(clsf, data, labels, cv=cv)
    cross_val_mean = cross_val.mean() * 100
    cross_val_std = cross_val.std() * 100
    return round(cross_val_mean, 3), round(cross_val_std, 3)


def train_and_validate_model(model, train, train_labels, test, test_labels, parameters=None):
    
    if parameters is not None:
        model = model(**parameters)
    else:
        model = model()
        
    model.fit(train, train_labels)
    train_valid = cross_val_mean_std(model, train, train_labels)
    test_valid = cross_val_mean_std(model, test, test_labels)
        
    res_of_valid = {"train_mean": train_valid[0], "train_std": train_valid[1], "test_mean":  test_valid[0],  "test_std":  test_valid[1]}
    return res_of_valid, model


def create_table_with_scores(res_of_valid, postfix=""):
    if not hasattr(res_of_valid["test_std"], "len"):
        index = [0]
    else:
        index = list(res_of_valid["test_std"])

    table = pd.DataFrame({"Test mean score" + postfix:  res_of_valid["test_mean"],
                          "Test std score" + postfix:   res_of_valid["test_std"],
                          "Train mean score" + postfix: res_of_valid["train_mean"],
                          "Train std score" + postfix:  res_of_valid["train_std"]}, 
                          index=index)
    return table

def table_of_results(model_results, model_names=None, col_sort_by=None):
    res = model_results[0]
    for i in model_results[1:]:
        res = res.append(i)
    if model_names is not None:
        names = []
        for i, j in enumerate(model_names):
            names += [j] * len(model_results[i])
        res["Model name"] = names
    if col_sort_by is not None:
        sort_by = res.columns[col_sort_by]
        res = res.sort_values(by=sort_by, ascending=False)
    res = res.reset_index(drop=True)
    return res

def tuning_models(model, params, train, train_labels, 
                                 test, test_labels, postfix="", iterations=50, cv=5):
    
    model_1 = model()
    random_search = RandomizedSearchCV(model_1, params, iterations, scoring='accuracy', cv=cv)
    random_search.fit(train, train_labels)
    
    parameter_set = []
    mean_test_scores = list(random_search.cv_results_['mean_test_score'])
    for i in sorted(mean_test_scores, reverse=True):
        if i > np.mean(mean_test_scores):
            parameter_set.append(random_search.cv_results_["params"][mean_test_scores.index(i)])
        
    params_set_updated = []
    for i in parameter_set:
        if i not in params_set_updated:
            params_set_updated.append(i)
    
    results = []
    for i in params_set_updated:
        res_of_valid, res_model = train_and_validate_model(model, train, train_labels, test, test_labels, parameters=i)
        print(i)
        print(res_of_valid)
        print(confusion_matrix(test_labels,res_model.predict(test)))
        results.append(create_table_with_scores(res_of_valid, postfix))
    results_table = table_of_results(results)
    
    return results_table


#split test and training sets and labeled sets

FEATURES = slice(0,-2, 1)
train, test, train_labels, test_labels = train_test_split(data[data.columns[FEATURES]], 
                                                          data[data.columns[-2:]], 
                                                          test_size=0.25, random_state=3)

b_train_labels = np.array(train_labels)[:, 1]
b_test_labels = np.array(test_labels)[:, 1]

train_labels = np.array(train_labels)[:, 0].astype(int)
test_labels = np.array(test_labels)[:, 0].astype(int)


#normalize
scalers = StandardScaler()
train = scalers.fit_transform(train)
test = scalers.fit_transform(test)




#do inital test
classifiers_scores = []
b_classifiers_scores = []

classifiers_importance = []

for i, clsf in enumerate(classifiers):
    t = [0, 0]
    
    res_of_valid, t[0] = train_and_validate_model(clsf, train, train_labels, test, test_labels)
    b_res_of_valid, t[1] = train_and_validate_model(clsf, train, b_train_labels, test, b_test_labels)
    
    classifiers_importance.append(t)
    
    classifiers_scores.append(create_table_with_scores(res_of_valid, " ('quality')"))
    b_classifiers_scores.append(create_table_with_scores(b_res_of_valid, " ('bin_quality')"))
    
classifiers_scores = table_of_results(classifiers_scores, model_names, 0)
b_classifiers_scores = table_of_results(b_classifiers_scores, model_names, 0)


# focus on the 3 stasrt with SVC with both binary and numerically lableled 
params = {"kernel": ["rbf", "poly", "linear", "sigmoid"],
          "C": np.arange(0.1, 1.5, 0.1), 
          "gamma": list(np.arange(0.1, 1.5, 0.1)) + ["auto"],
          "probability": [True, False],
          "shrinking": [True, False]}
print("start svc_res")
svc_res = tuning_models(SVC, params, train, train_labels, 
                        test, test_labels, " ('quality')", 100)
print("end svc_res")
print("start b_svc_res")
b_svc_res = tuning_models(SVC, params, train, b_train_labels, 
                          test, b_test_labels, " ('bin_quality')", 100)
print("end b_svc_res")



#start ExtraTreesClassifier and RandomForestClassifier
   
params = {"n_estimators": np.arange(1, 500, 2),
          "max_depth": list(np.arange(2, 100, 2)) + [None],
          "min_samples_leaf": np.arange(1, 20, 1),
          "min_samples_split": np.arange(2, 20, 2),
          "max_features": ["auto", "log2", None]}

print("start extra_res")
extra_res = tuning_models(ExtraTreesClassifier, params, train, train_labels, 
                          test, test_labels, " ('quality')", 100)
print("end extra_res")

print("start b_extra_res")
b_extra_res = tuning_models(ExtraTreesClassifier, params, train, b_train_labels, 
                            test, b_test_labels, " ('bin_quality')", 100)
print("end b_extra_res")
print("start forest_res")
forest_res = tuning_models(RandomForestClassifier, params, train, train_labels, 
                           test, test_labels, " ('quality')", 100)
print("end forest_res")
print("start b_forest_res")
b_forest_res = tuning_models(RandomForestClassifier, params, train, b_train_labels, 
                             test, b_test_labels, " ('bin_quality')", 100)
print("end b_forest_res")


#table of best models, both binary and numeric
all_results = table_of_results([svc_res, extra_res, forest_res], 
                               ["SVC", "ExtraTrees", "RandomForest"], 0)
all_results.head(10)

b_all_results = table_of_results([b_svc_res, b_extra_res, b_forest_res], 
                                 ["SVC", "ExtraTrees", "RandomForest"], 0)
b_all_results.head(10)




#best parameters on the 3 models
params_of_b_best_random = {"n_estimators": 159,
          "max_depth": 14,
          "min_samples_leaf": 2,
          "min_samples_split": 4,
          "max_features": "log2"}
params_of_b_most_tgood_extra = {"n_estimators": 359,
          "max_depth": 52,
          "min_samples_leaf": 1,
          "min_samples_split": 2,
          "max_features": None}

params_of_b_best_and_tgood_svc = {"kernel": "rbf",
          "C": 1.1, 
          "gamma": 0.6,
          "probability": False,
          "shrinking": True}


res_of_b_best_random, model_of_b_best_random = train_and_validate_model(RandomForestClassifier, train, b_train_labels, test, b_test_labels, params_of_b_best_random)

res_of_b_most_tgood_extra, model_of_b_most_tgood_extra = train_and_validate_model(ExtraTreesClassifier, train, b_train_labels, test, b_test_labels, params_of_b_most_tgood_extra)

res_of_b_best_svc, model_of_b_best_svc = train_and_validate_model(SVC, train, b_train_labels, test, b_test_labels, params_of_b_best_and_tgood_svc)


res_of_best_random, model_of_best_random = train_and_validate_model(RandomForestClassifier, train, train_labels, test, test_labels, params_of_b_best_random)
res_of_most_tgood_extra, model_of_most_tgood_extra = train_and_validate_model(ExtraTreesClassifier, train, train_labels, test, test_labels, params_of_b_most_tgood_extra)
res_of_best_svc, model_of_best_svc = train_and_validate_model(SVC, train, train_labels, test, test_labels, params_of_b_best_and_tgood_svc)


print("\n\nThe best test cv score by means of RandomForestClassifier")
print(params_of_b_best_random)
print(res_of_b_best_random)
print("Results in this confusion matrix")
print(confusion_matrix(b_test_labels,model_of_b_best_random.predict(test)))
print(classification_report(b_test_labels, model_of_b_best_random.predict(test)))


print("\n\nThe best in terms of true identification of good by means of ExtraTreesClassifier")
print(params_of_b_most_tgood_extra)
print(res_of_b_most_tgood_extra)
print("Results in this confusion matrix")
print(confusion_matrix(b_test_labels,model_of_b_most_tgood_extra.predict(test)))
print(classification_report(b_test_labels, model_of_b_most_tgood_extra.predict(test)))

print("\n\nThe best test cv score and best in terms of true identification of good by means of SVC")
print(params_of_b_best_and_tgood_svc)
print(res_of_b_best_svc)
print("Results in this confusion matrix")
print(confusion_matrix(b_test_labels,model_of_b_best_svc.predict(test)))
print(classification_report(b_test_labels, model_of_b_best_svc.predict(test)))

print("numeric")
print("\n\nBest model of numeric labeled data by means of RandomForestClassifier")
print(params_of_b_best_random)
print(res_of_best_random)
print("Results in this confusion matrix")
print(confusion_matrix(test_labels,model_of_best_random.predict(test)))
print(mean_squared_error(test_labels, model_of_best_random.predict(test), squared=False))
print(classification_report(test_labels, model_of_best_random.predict(test)))

print("\n\nBest model of numeric labeled data by means of ExtraTreesClassifier")
print(params_of_b_most_tgood_extra)
print(res_of_most_tgood_extra)
print("Results in this confusion matrix")
print(confusion_matrix(test_labels,model_of_most_tgood_extra.predict(test)))
print(mean_squared_error(test_labels, model_of_most_tgood_extra.predict(test), squared=False))
print(classification_report(test_labels, model_of_most_tgood_extra.predict(test)))

print("\n\nBest model of numeric labeled data by means of SVC")
print(params_of_b_best_and_tgood_svc)
print(res_of_best_svc)
print("Results in this confusion matrix")
print(confusion_matrix(test_labels,model_of_best_svc.predict(test)))
print(mean_squared_error(test_labels, model_of_best_svc.predict(test), squared=False))
print(classification_report(test_labels, model_of_best_svc.predict(test)))
