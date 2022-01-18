from estimators.liner_regression import LogisticRegression as ILogigsticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datasets.regression_test import regression_test_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from tools import print_table
import pandas as pd
import loss

df = regression_test_data()

X = df.iloc[:, :-1]
y = pd.DataFrame(df.iloc[:, -1])
X = pd.DataFrame(StandardScaler().fit_transform(X, y))

print('Objects')
print_table(X)
print('Targets')
print_table(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sk_regression = LinearRegression().fit(X_train, y_train)
i_regression = iModelsLinearRegression(alpha=0.0001, epoch=1000).fit(X_train, y_train)
print(loss.sigmoid(i_regression.w, i_regression._add_extra_parameter(X)))
sk_pred = pd.DataFrame(sk_regression.predict(X_test))
i_pred = pd.DataFrame(i_regression.predict(X_test))

print("SK mse(sk): {}".format(mean_squared_error(y_test, sk_pred.to_numpy())))
print("I mse(sk): {}".format(mean_squared_error(y_test, i_pred.to_numpy())))
