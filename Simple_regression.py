import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Load data from CSV
data = pd.read_csv('home_dataset.csv')

#Extract x and y axis from the data
house_sizes = data['HouseSize'].values
house_prices = data['HousePrice'].values
x = np.array(house_sizes).reshape(-1, 1) 
y = np.array(house_prices).reshape(-1, 1) 

#Split the data into testing and training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42) 
  
#Train the data using LinearRegression from Scikit
regr = LinearRegression()

#Find the R^2 and pred from trained model
regr.fit(x_train, y_train) 
print(regr.score(x_test, y_test)) 
pred = regr.predict(x_test)

# Visualize the predictions
plt.scatter(x_test, y_test, marker='o', color='blue', label='Actual Prices')
plt.plot(x_test, pred, color='red', linewidth=2, label='Predicted Prices')
plt.title('Dumbo Property Price Prediction with Linear Regression')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price (millions $)')
plt.legend()
plt.show()