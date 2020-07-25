############ Data Preprocessing ############
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
ts=time.time()
import datetime
from matplotlib import style 
style.use('ggplot')

epoch =1000

from tkinter import Tk
from tkinter.filedialog import askopenfilename
ftypes = [("Comma Separated Values","*.csv")]
root=Tk()
root.fileName=askopenfilename(filetypes=ftypes)
#print(root.fileName)
file=root.fileName
root.destroy()
# Importing the dataset
ds = pd.read_csv(file)

dataset = ds.iloc[:, [1,4]].values

X = ds.iloc[:, 1].values
y = ds.iloc[:, 4].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
 
scaler  = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

X = dataset_scaled[:, 0]
y = dataset_scaled[:, 1]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Sizes of train_ds, test_ds
dataset_sz = X.shape[0]
train_sz = X_train.shape[0]
test_sz = X_test.shape[0]


############ Building the ANN ############
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))
regressor.add(Dropout(.2))

# Adding the first hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))

# Adding the second hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))

# Adding the third hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))

# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = epoch)


############ Save & load Trained Model ############
# Save & load Trained Model
from keras.models import load_model

# Save Trained Model
regressor.save('test1.h5')

# deletes the existing model
del regressor

# load Trained Model
regressor = load_model('test1.h5')

############ Predict & Test the Model ############
real_stock_price = np.array(X_test)
inputs = real_stock_price
predicted_stock_price = regressor.predict(inputs)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = real_stock_price
dataset_test_total['predicted'] = predicted_stock_price

# real data price VS. predicted price
predicted_stock_price = scaler.inverse_transform(dataset_test_total) 

# Calc difference between real data price and predicted price
toler_rate = np.zeros(test_sz)

for i in range(0, test_sz):
    toler_rate[i] = abs(predicted_stock_price[i, 0] - predicted_stock_price[i, 1])


significance = 5.0

# Wrong predicted count
err_cnt = 0
for i in range(0, test_sz):
    if abs(predicted_stock_price[i, 0] - predicted_stock_price[i, 1]) <= significance/100 * predicted_stock_price[i, 0] :
        pass
    else:
        err_cnt +=1



# Calc MSE
mse = 0.0
for i in range(0, test_sz):
    mse += (predicted_stock_price[i, 0] - predicted_stock_price[i, 1])**2

mse /= test_sz

st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("Training started at :",st)
ts=time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("Training ended at :",st)



all_real_stock_price = np.array(y)
inputs = np.array(X)
#inputs = np.reshape(inputs, (dataset_sz, 1, 1))
all_predicted_stock_price = regressor.predict(inputs)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = all_real_stock_price
dataset_test_total['predicted'] = all_predicted_stock_price

# real test data price VS. predicted price
stock_price_predicted_real = scaler.inverse_transform(dataset_test_total) 

# Visualising the results
plt.plot(stock_price_predicted_real[:, 0], color = 'red', label = 'Real Stock Price')
plt.plot(stock_price_predicted_real[:, 1], color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Index')
plt.ylabel('Stock Price')
#plt.savefig('stock_price_predicted.png')
plt.legend()
plt.show()



#plt.plot(stock_price_predicted_real[:, 0], color = 'red', label = 'Real Stock Price')
#plt.title('Stock Prices')
#plt.xlabel('Index')
#plt.ylabel('Stock Price')
#plt.savefig('Real_stock_price5.png')
#plt.show()


#plt.plot(stock_price_predicted_real[:, 1], color = 'blue', label = 'Predicted Stock Price')
#plt.title('Stock Price Prediction')
#plt.xlabel('Index')
#plt.ylabel('Stock Price')
#plt.savefig('Predicted_stock_price5.png')
#plt.show()