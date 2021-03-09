import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


all_registrations = pd.read_csv('all_registrations.csv', sep=' ', names=['day', 'my_date', 'my_time', 'user_id', 'event_type', 'birth_year', 'phone_type', 'location', 'source'])
all_free_tree = pd.read_csv('all_free_tree.csv', sep=' ', names=['day', 'my_date', 'user_id', 'event_type'])
all_super_tree = pd.read_csv('all_super_tree.csv', sep=' ', names=['day', 'my_date', 'user_id', 'event_type'])


all_free_tree['my_date'] = pd.to_datetime(all_free_tree['my_date'])
all_super_tree['my_date'] = pd.to_datetime(all_super_tree['my_date'])
all_registrations['my_date'] = pd.to_datetime(all_registrations['my_date'])


# Quick check for dataframes
# all_registrations.head()
# all_free_tree.head()
# all_super_tree.head()


free_tree = all_free_tree.groupby('user_id')['event_type'].count()
super_tree = all_super_tree.groupby('user_id')['event_type'].count()

free_super = pd.merge(free_tree,super_tree, on='user_id',how='outer' , suffixes=('_free', '_super'))
free_super.reset_index(level=0, inplace=True)

# Quick check
free_super.head()

# Missing values
free_super.fillna(0, inplace=True)

big_table = pd.merge(all_registrations, free_super, on='user_id')
big_table = big_table[['user_id', 'phone_type', 'location', 'source', 'event_type_free', 'event_type_super']]
# big_table.head()



# DATA CLEANING

# known_values & unknown_values

known_values = big_table[(big_table.phone_type == 'android') | (big_table.phone_type == 'ios')]
unknown_values = big_table[big_table.phone_type == 'error']

known_values = pd.get_dummies(known_values,columns=['location', 'source']) 
# because there is no ordering between the feature values (I think)
known_values['phone_type'] = known_values.phone_type.map({'ios':0, 'android':1})

# Quick check 
# known_values.head()

# Unknown values
unknown_values = pd.get_dummies(unknown_values, columns=['location', 'source']) 
# unknown_values.head()



# MACHINE LEARNING

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

# X,Y variables and train_test_split
X = known_values.drop(['user_id', 'phone_type'], axis =1)
y = known_values['phone_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) # stratify=y

# Pipeline
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_2 = StratifiedKFold(n_splits=10, shuffle=True,random_state=42)

param_grid = [
{'classifier': [LogisticRegression()], 'preprocessing': [StandardScaler(), None],
'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
{'classifier': [RandomForestClassifier(n_estimators=100)],
'preprocessing': [None], 'classifier__max_features': [3, 4, 5]}]  # max_features=sqrt(n_features) -> np.sqrt(11)

pipe = Pipeline([('preprocessing', StandardScaler()),
                 ('classifier', LogisticRegression(max_iter=10000))])



# Cross-validation + GridSearchCV + PipeLine
scores = cross_val_score(GridSearchCV(pipe, param_grid, cv=5), X,y, cv=cv_2, scoring='roc_auc')
# scoring = 'roc_auc'
# scoring = 'accuracy'
scores.mean()


# GridSearchCV + PipeLine
grid = GridSearchCV(pipe, param_grid, cv=cv_2, scoring='roc_auc')
grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_)) # Best params: {'classifier': RandomForestClassifier(max_features=5), 'classifier__max_features': 5, 'preprocessing': None}
print("Best cross-validation score: {:.2f}".format(grid.best_score_)) # Best cross-validation score: 0.92
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test))) # Test-set score: 0.92



# Rebuilding the model with the found parameters

rf = RandomForestClassifier(n_estimators=100, max_features=5)
rf.fit(X_train,y_train)
rf.score(X_test, y_test) 
rf.score(X_test, y_test) # 0.8503

pred_rf = rf.predict(X_test)


# Evaluation metrics

from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, auc, roc_curve, precision_recall_curve


print(classification_report(pred_rf, y_test)) # accuracy = 0.85 


# Visualizing feature importances

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh')


# Visualizing Precision recall curve

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(precision_rf, recall_rf, label="random forest")
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k', markersize=10, label="threshold 0.5 random forest", fillstyle="none", mew=2)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")

# Avarage precision

from sklearn.metrics import average_precision_score
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
print("Average precision of random forest: {:.3f}".format(ap_rf))
# Average precision of random forest: 0.979


# Visualizing Receiver operating characteristics (ROC)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test,rf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
label="threshold zero", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)

# AUC
from sklearn.metrics import roc_auc_score
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print("AUC for Random Forest: {:.3f}".format(rf_auc))
# AUC for Random Forest: 0.917




# PREDICTING THE UNKNOWN VALUES

# Quick check for the dataframe
unknown_values.head()

X_unknown = unknown_values.drop(['user_id', 'phone_type'], axis=1)
# X_unknown.head()

rf = RandomForestClassifier(n_estimators=100, max_features=5)
rf.fit(X,y)
pred_X_unknown = rf.predict(X_unknown)

pred_X_unknown.tolist().count(1)
pred_X_unknown.tolist().count(0)

unknown_values['phone_type'] = pred_X_unknown
unknown_values['phone_type'].value_counts().plot(kind='bar')




# Unbalanced dataset


# Quick check for the dataframe
known_values.head()
known_values['phone_type'].value_counts() # unbalanced dataset -> 1: 29274; 0: 8047


from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

from collections import Counter
print(sorted(Counter(y_resampled).items()))


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_features=5)
rf.fit(X_train, y_train)
pred_rf_oversampled = rf.predict(X_test)

print(classification_report(y_test, pred_rf_oversampled)) # accuracy =  0.88 

pred_oversampled_X_unknown = rf.predict(X_unknown)


pred_oversampled_X_unknown.tolist().count(1) # 20792
pred_oversampled_X_unknown.tolist().count(0) # 5312

# because there is a pike in birth_year it can be interesting to see that
# by creating groups

