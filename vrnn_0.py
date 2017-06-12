# Code adapted from:
# http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
# and 
# https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction
# Note: this is just a vanilla RNN, no LSTM components
# F. Burkholder 9 June 2017
# for Python 3 (change print statements for Python 2)

import numpy as np                  #use numpy for arrays and lin. algebra
import pdb
import matplotlib.pyplot as plt
np.random.seed(1)                   #sets the random seed for reproducible results



def load_data(filename, cycles, pnt_step, train_size, seq_len):
    ''' Loads 5000 row sin wave form.  Each cycle is 100 rows and there are 50
        cycles.
        
        Parameters:
                filename: name of file (sinwave.csv)
                cycles: number of cycles to keep
                pnt_step: if 1, keep every pt in cycle, 2 keeps every other pt, etc.
                train_size: fraction of rows to train data on (sets train-test split)
                seq_len: the number of pts in the sequence during train and test
    '''
    last_row = cycles * 100
    data = []
    with open(filename, 'r') as f:
       for line in f:
            data.append(float(line))
    data = data[0:last_row:pnt_step]

    sequence_length = seq_len + 1 # to include last point as target
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    result = np.array(result)
    row = int(round(train_size * result.shape[0]))
    train = result[:row, :]  # everything up to row is in the train set
    np.random.shuffle(train) # shuffle these rows
    x_train = train[:, :-1] # in each row, the train data doesn't include last pt
    y_train = train[:, -1]  # in each row, the target is the last pt
    x_test = result[row:, :-1] # same for test
    y_test = result[row:, -1]  # same for test

    return x_train, y_train, x_test, y_test

# neuron activation function
def activation(x, f='sigmoid'):
    if f == 'tanh':
        return np.tanh(x)
    else:
        return 1/(1 + np.exp(-x)) #sigmoid

# derivative of activation function
def activation_deriv(x, f='sigmoid'):
    if f == 'tanh':
        return 1 - np.tanh(x)**2
    else:
        return x*(1 - x)        #not correct, but converges faster


# neural net architecture
input_dim = 8                       #8 length of input sequence
hidden_dim = 20                     #20 orig, number of neurons in the hidden layer
output_dim = 1                      #output 

# training parameters
num_epochs = 5000 #number of passes through the training set
alpha = 0.1     #learning rate

# initialize neural network weights - uniform from -1 to 1
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1  # weights from input to hidden layer
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # weights hidden layer to output
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # weights hidden layer to itself

# weight update arrays
synapse_0_update = np.zeros_like(synapse_0)     #same shape as synapse_0
synapse_1_update = np.zeros_like(synapse_1)     #same shape as synapse_1
synapse_h_update = np.zeros_like(synapse_h)     #same shape as synapse_h

# read in data
X_train, y_train, X_test, y_test = load_data(filename='sinwave.csv', cycles=50, 
                                             pnt_step=2, train_size=0.8, seq_len=input_dim)

# training
# initializations
print("Training\n")
train_error = list() # error associated 
layer_2_deltas = list() # error between output and hidden layer (layer 1)
layer_1_values = list() # layer 1 values 
layer_2_values = list() # to store predictions
y_values = list() # to store true targets
layer_1_values.append(np.zeros((1, hidden_dim))) # layer 1 initial values
layer_1_delta_fut = np.zeros(hidden_dim)
#overallError = 0  # initialize loss for this epoch
dp = 0
for e in range(num_epochs):
    #pdb.set_trace() 
    epoch_error = 0
    for X, y in zip(X_train, y_train):
        X, y = X.reshape((1, X.shape[0])), y.reshape((1,1))
        # feed forward
        layer_1_previous = layer_1_values[-1] # the last row of layer_1_values
        layer_1 = activation(np.dot(X,synapse_0) + np.dot(layer_1_previous,synapse_h), 'tanh')  # calculate layer 1 values
        layer_1_values.append(layer_1)
        layer_2 = activation(np.dot(layer_1,synapse_1), 'tanh') #layer 2 prediction (y_pred) 
        layer_2_error = y - layer_2 # how far off is prediction?
        epoch_error += np.abs(layer_2_error[0][0])
        # back propogation
        layer_2_delta = (layer_2_error)*activation_deriv(layer_2, 'tanh')
        layer_1_delta = (np.dot(layer_1_delta_fut, synapse_h.T) + np.dot(layer_2_delta, synapse_1.T)) * activation_deriv(layer_1, 'tanh')
        #layer_1_delta_fut = np.copy(layer_1_delta[0])
        
        synapse_0_update += np.dot(X.T, layer_1_delta)
        synapse_1_update += np.dot(layer_1.T, layer_2_delta)
        synapse_h_update += np.dot(layer_1_previous.T, layer_1_delta)
        
        synapse_0 += alpha * synapse_0_update
        synapse_1 += alpha * synapse_1_update
        synapse_h += alpha * synapse_h_update

        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0
    avg_epoch_err = epoch_error/X_train.shape[0]
    train_error.append(avg_epoch_err)
    if e % 100 == 0:
        print("Epoch: {0}, train error: {1:0.3f}".format(e, avg_epoch_err))
epochs = np.arange(num_epochs)
plt.plot(epochs, train_error)
plt.show()
#

#Testing
#printing only one value ahead
#print("Testing\n")
layer_1_values = list() # layer 1 values 
layer_1_values.append(np.zeros((1, hidden_dim))) # layer 1 initial values
y_pred = list()
for X in X_test:
    X = X.reshape((1, X.shape[0]))
    # feed forward
    layer_1_previous = layer_1_values[-1] # the last row of layer_1_values
    layer_1 = activation(np.dot(X,synapse_0) + np.dot(layer_1_previous,synapse_h), 'tanh')  # calculate layer 1 values
    layer_1_values.append(layer_1)
    layer_2 = activation(np.dot(layer_1,synapse_1), 'tanh') #layer 2 prediction (y_pred) 
    y_pred.append(layer_2[0])

#Testing
#seeding with the first 5 values and then going from predictions from then on
#print("Testing\n")

layer_1_values = list() # layer 1 values 
layer_1_values.append(np.zeros((1, hidden_dim))) # layer 1 initial values
y_pred_full = list()
for i in range(X_test.shape[0]):
    if i == 0:
        X = X_test[0]
        X = X.reshape((1, X.shape[0]))
    # feed forward
    layer_1_previous = layer_1_values[-1] # the last row of layer_1_values
    layer_1 = activation(np.dot(X,synapse_0) + np.dot(layer_1_previous,synapse_h), 'tanh')  # calculate layer 1 values
    layer_1_values.append(layer_1)
    layer_2 = activation(np.dot(layer_1,synapse_1), 'tanh') #layer 2 prediction (y_pred) 
    y_pred_full.append(layer_2[0])
    X = np.append(X[:,1:], layer_2[0]).reshape((1,input_dim))

num_dp = X_test.shape[0]
dp = np.arange(0,num_dp)
plt.plot(dp, y_test, '-b')
plt.plot(dp, y_pred, '-r')
plt.plot(dp, y_pred_full, '-g')
plt.show()




#
#        
#        # store hidden layer so we can use it in the next timestep, also useful for gradient descent
#        layer_1_values.append(np.copy(layer_1))
#        # here store layer 2 values - can use layer 2 with y values to get error
#        layer_2_values.append(np.copy(layer_2))
#        y_values.append(np.copy(y))
#
#    print('The error for the epoch was {0}'.format(overallError))
#
#    layer_1_delta_fut = np.zeros(hidden_dim)
#    i = 0 
#    for X, y in zip(X_train, y_train):
#        layer_1 = layer_1_values[i + 1]
#        prev_layer_1 = layer_1_values[i]
#        
#        # error at output layer
#        #layer_2_delta = layer_2_deltas[-position-1]
#        
#        # my attempt to do here 
#        l2 = layer_2_values[i] 
#        layer_2_error = y_values[i] - l2
#        layer_2_delta = (layer_2_error)*activation_deriv(l2, 'tanh')
#
#        # error at hidden layer
#        layer_1_delta = (np.dot(layer_1_delta_fut, synapse_h.T) + np.dot(layer_2_delta, synapse_1.T)) * activation_deriv(layer_1, 'tanh')
#
#        # let's update all our weights so we can try again
#        synapse_1_update += np.dot(layer_1.T, layer_2_delta)
#        synapse_h_update += np.dot(prev_layer_1.T, layer_1_delta)
#        synapse_0_update += np.dot(X.T, layer_1_delta)
#        
#        layer_1_delta_fut = layer_1_delta
#    
#
#    synapse_0 += alpha * synapse_0_update
#    synapse_1 += alpha * synapse_1_update
#    synapse_h += alpha * synapse_h_update
#
#    # set update matrices values to zero 
#    synapse_0_update *= 0
#    synapse_1_update *= 0
#    synapse_h_update *= 0
#    
#    # print out progress
#    if(j % 1000 == 0):
#        print("Error:" + str(overallError))
#        print("Pred:" + str(d_bin))
#        print("True:" + str(c_bin))
#        out = 0
#        for index,x in enumerate(reversed(d_bin)):
#            out += x*pow(2,index)
#        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
#        print("------------")
#
#        
