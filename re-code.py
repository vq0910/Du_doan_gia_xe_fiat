#import thu vien
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
data = pd.read_csv("data1.csv")
X = data.drop('price', axis = 1)
y = data.price.values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=False)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
regressor = LinearRegression()
regressor.fit(X_train_poly,y_train)
# w_0=regressor.intercept_
# w_1=regressor.coef_
# print(w_0 ,w_1 )
y_train_pred = regressor.predict(X_train_poly)
y_test_pred = regressor.predict(X_test_poly)
errr=abs(y_test-y_test_pred)
for i in range (len(y_test)):
    print('Giá xe thực tế  :'+ str(y_test[i] ) +' USD.  '+'Giá xe dự đoán  :'+ str(y_test_pred[i]) + ' USD.  '+'Chênh lệch: '+ str(errr[i]) + ' USD.')
plt.scatter(y_test, y_test_pred)
plt.scatter(y_test, y_test)
plt.show()



