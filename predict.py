############ Data Preprocessing ############
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import style 
style.use('ggplot')

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
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#print(X_test)


# Sizes of train_ds, test_ds
dataset_sz = X.shape[0]
#train_sz = X.shape[0]
test_sz = X.shape[0]


import keras
from keras.models import Sequential
from keras.models import load_model
# load Trained Model
regressor = load_model('model.h5')

############ Visualizing the results ############
all_real_stock_price = np.array(y)
inputs = np.array(X)
#inputs = np.reshape(inputs, (dataset_sz, 1, 1))
all_predicted_stock_price = regressor.predict(inputs)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = all_real_stock_price
dataset_test_total['predicted'] = all_predicted_stock_price

# real test data price VS. predicted price
predicted_stock_price = scaler.inverse_transform(dataset_test_total) 

# Visualising the results
plt.plot(predicted_stock_price[:, 0], color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price[:, 1], color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Index')
plt.ylabel('Stock Price')
#plt.savefig('stock_price_predicted_real5.png')
plt.legend()
plt.show()


#plt.plot(predicted_stock_price[:, 0], color = 'red', label = 'Real Stock Price')
#plt.title('Stock Prices')
#plt.xlabel('Index')
#plt.ylabel('Stock Price')
#plt.savefig('Real_stock_price5.png')
#plt.show()


#plt.plot(predicted_stock_price[:, 1], color = 'blue', label = 'Predicted Stock Price')
#plt.title('Stock Price Prediction')
#plt.xlabel('Index')
#plt.ylabel('Stock Price')
#plt.savefig('Predicted_stock_price5.png')
#plt.show()

significane = 10.0

# Wrong predicted count
err_cnt = 0
for i in range(0, test_sz):
    if abs(predicted_stock_price[i, 0] - predicted_stock_price[i, 1]) <= significane/100 * predicted_stock_price[i, 0] :
        pass
    else:
        err_cnt +=1
#print("Error count:",err_cnt)

print("Error percentage with ",significane," % significance :",err_cnt/test_sz*100)
import math

# Calc MSE
mse = 0.0
for i in range(0, test_sz):
    mse += (predicted_stock_price[i, 0] - predicted_stock_price[i, 1])**2
    

mse /= test_sz
print("Mean Square error:",mse)
nxt=predicted_stock_price[test_sz-1,1]
print("The next stock price will be: ",nxt)