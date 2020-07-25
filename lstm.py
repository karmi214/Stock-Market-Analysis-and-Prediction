import copy, numpy as np
import pandas as pd
#ds = pd.read_csv('TSLA.csv')
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
ds = pd.read_csv('Sp500.csv')


# input variables
alpha = 0.1
input_dim = 1
hidden_dim1 = 400
hidden_dim2 = 400
hidden_dim3 = 400
output_dim = 1


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim1)) - 1
synapse_1 = 2*np.random.random((hidden_dim3,output_dim)) - 1
synapse_h1 = 2*np.random.random((hidden_dim1,hidden_dim2)) - 1
synapse_h2 = 2*np.random.random((hidden_dim2,hidden_dim3)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h1_update = np.zeros_like(synapse_h1)
synapse_h2_update = np.zeros_like(synapse_h2)
synapse_h3_update = np.zeros_like(synapse_h3)

# training logic
for j in range(1000):
    
    a=ds.Close
    
    predicted = np.zeros_like(c)

    overallError = 0
    layer_4_deltas = list()
    layer_2_deltas = list() 
    layer_3_deltas = list()
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim1))
    
    # moving along the positions in the binary encoding
    for position in range(ds.index):
        
        # generate input and output
        X = ds.Close
        y = np.array(ds.Close).T

        # hidden layer 1
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h1))

        # hidden layer 2
        layer_2 = sigmoid(np.dot(X,synapse_h1) + np.dot(layer_2_values[-1],synapse_h2))

        # hidden layer 3
        layer_3 = sigmoid(np.dot(X,synapse_h2) + np.dot(layer_3_values[-1],synapse_h3))


        # output layer
        layer_oo = sigmoid(np.dot(layer_3,synapse_h2))
        
        layer_4_error = y - layer_oo
        layer_4_deltas.append((layer_4_error)*sigmoid_output_to_derivative(layer_2))
        layer_3_error = y - layer_3
        layer_3_deltas.append((layer_3_error)*sigmoid_output_to_derivative(layer_2)) 
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        
        
        layer_1_values.append(copy.deepcopy(layer_1))
        layer_2_values.append(copy.deepcopy(layer_2))
        layer_3_values.append(copy.deepcopy(layer_3))
    
    future_layer_1_delta = np.zeros(hidden_dim1)
    future_layer_2_delta = np.zeros(hidden_dim2)
    future_layer_3_delta = np.zeros(hidden_dim3)
    
    for position in range(ds.index):
        
        X = np.array(ds.Close)
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        
        layer_1_delta = (future_layer_1_delta.dot(synapse_h1.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        layer_2_delta = (future_layer_2_delta.dot(synapse_h2.T) + layer_3_delta.dot(synapse_2.T)) * sigmoid_output_to_derivative(layer_2)
        layer_3_delta = (future_layer_3_delta.dot(synapse_h3.T) + layer_3_delta.dot(synapse_3.T)) * sigmoid_output_to_derivative(layer_3)

        
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h1_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_h2_update += np.atleast_2d(prev_layer_1).T.dot(layer_2_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h1_update *= 0
    synapse_h2_update *= 0
    
    # print out progress
    #if(j % 1000 == 0):
    #    print "Error:" + str(overallError)
    #    print "Pred:" + str(predicted)
    #    print "True:" + str(a)
    #    out = 0

model=pd.DataFrame(synapse_0 ,synapse_1 ,synapse_h1 ,synapse_h2,synapse_h3)
model.save_to('model.h5')        

        
