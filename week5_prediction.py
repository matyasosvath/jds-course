import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns

all_free_tree = pd.read_csv('all_free_tree.csv', sep=' ', names=['day', 'my_date', 'user_id', 'event_type'], parse_dates=['my_date'])

all_free_tree.sort_values(by='my_date', axis=0, inplace=True)
all_free_tree.reset_index(drop=True, inplace=True)
# all_free_tree.head()

free_tree_sends_per_day = all_free_tree.groupby('my_date')['event_type'].count().sort_values()
# free_tree_sends_per_day.head()

free_tree_sends_per_day = free_tree_sends_per_day.reset_index(level=(0))
free_tree_sends_per_day = free_tree_sends_per_day.reset_index(level=(0))
free_tree_sends_per_day = free_tree_sends_per_day.rename(columns = {'index': 'day_num'})
# free_tree_sends_per_day.head()


# Visualizing

# plt.figure(figsize=(10,10))
sns.regplot(x='day_num',y='event_type',data=free_tree_sends_per_day, order=1)
# plt.figure(figsize=(10,10))
sns.regplot(x='day_num',y='event_type',data=free_tree_sends_per_day, order=3)




free_tree_sends_per_day = all_free_tree.groupby('my_date')['event_type'].count().sort_values()
# free_tree_sends_per_day.head()

y = free_tree_sends_per_day.values
X = np.arange(0,186) 

y = y.reshape(-1,1)
X = X.reshape(-1,1)

next_4_weeks = np.arange(186,215).reshape(-1,1) # 187 helyett 186 lett
# next_4_weeks[:5]


# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
                 ("scaler", StandardScaler()),
                 ("poly", PolynomialFeatures(degree=3)) , 
                 ("linreg", LinearRegression())])

pipe.fit(X, y)
pred_poly_pipe = pipe.predict(next_4_weeks)
pipe.predict([[186+28]]) # 18825.52669579


# Cross validation

from sklearn.model_selection import KFold, cross_val_score
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X,y, cv=cv)
scores.mean() # 0.9960419222453545


# Check Residuals

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3, include_bias=False)
# include polynomials to x ** 3

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly.get_feature_names()

linreg = LinearRegression().fit(X_train_poly, y_train)
linreg.score(X_test_poly, y_test)

pred_poly = linreg.predict(X_test_poly)

from sklearn.metrics import r2_score
r2_score(pred_poly, y_test)

sns.distplot((y_test-pred_poly),bins=30) # right skewed dist

