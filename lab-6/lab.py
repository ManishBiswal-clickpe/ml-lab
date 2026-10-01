
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Salary_Data.csv')


X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

X


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


len(X_Train), len(Y_Train), len(X_Test), len(Y_Test)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)


Y_Pred = regressor.predict(X_Test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
MSE = mean_squared_error(Y_Pred,Y_Test)
RMSE = pow(MSE,0.5)
print(f'MSE:{MSE}',f'RMSE:{RMSE}')
mae = mean_absolute_error(Y_Pred,Y_Test)
print(f'MAE: {mae}')
APE = np.abs((Y_Pred-Y_Test) / Y_Test)*100
MAPE = np.mean(APE)
print(f'MAPE: {MAPE}')
r2 = r2_score(Y_Test, Y_Pred)
print("R-squared:", r2)


plt.scatter(X_Train, Y_Train, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Salary vs Experience  (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()



plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Salary vs Experience  (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

residuals=Y_Test-Y_Pred
print('Test sum of squared residuals:',np.sum(residuals**2))
plt.figure();plt.scatter(Y_Pred,residuals);plt.axhline(0,color='black',linestyle='--');plt.xlabel('Predicted salary');plt.ylabel('Residual');plt.title('Linear regression test residuals');plt.show()