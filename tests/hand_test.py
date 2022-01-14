from estimators.liner_regression import DGLinearRegression as iModelsLinearRegression
from metrix.metrix import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datasets.regression_test import regression_test_data
from tools import print_table
import pandas as pd

df = regression_test_data()


X = df.iloc[:, :-1]
y = pd.DataFrame(df.iloc[:, -1])
print('Objects')
print_table(X)
print('Targets')
print_table(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sk_regression = LinearRegression().fit(X_train, y_train)
i_regression = iModelsLinearRegression().fit(X_train, y_train)

sk_pred = sk_regression.predict(X_test)
i_pred = i_regression.predict(X_test)
print(i_pred)
print("SK mse(sk): {}".format(mean_squared_error(y_test, sk_pred)))
print("I mse(sk): {}".format(mean_squared_error(y_test, i_pred)))
