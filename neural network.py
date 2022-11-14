import numpy as np
import pandas as pd


#Load the data
df_train = pd.read_csv(r".\mnist_train.csv")
df_test = pd.read_csv(r"C:.\mnist_test.csv")
df_train = pd.DataFrame(df_train).to_numpy().transpose() #we want every column to represent a picture together with its label
df_test = pd.DataFrame(df_test).to_numpy().transpose()
print(f"Training data shape: {df_train.shape}")
print(f"Testing data shape: {df_test.shape}")


#preprocess the data
df_train = df_train[:, np.random.permutation(df_train.shape[1])] #shuffle columns of the training data
X_train = df_train[1:,:]/255
Y_train = df_train[0,:]
X_trainsamples = []
Y_trainsamples = []
for i in range(60):
    X_trainsamples.append(X_train[:,1000*i:1000*i+1000])
    Y_trainsamples.append(Y_train[1000*i:1000*i+1000])
X_test = df_test[1:,:]/255
Y_test = df_test[0,:]


#initialize the network by specifying number of neurons in hidden layers and number of hidden layers
def init(n :int,m : int):
    W = []
    W.append(np.random.rand(n,784)-0.5)    #weights for input layer
    for i in range(m-1):                     #weights for hidden layers
        W.append(np.random.rand(n,n)-0.5)
    W.append(np.random.rand(10,n)-0.5)
    B = []
    for i in range(m):                     #weights for hidden layers
        B.append(np.zeros([n,1]))
    B.append(np.zeros([10,1]))             #weight for output layer
    return W, B, m


#define Sigmoid function
def sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)



#forward propagation
def forward_prop(m, W, B, X):
    Z=[]
    A=[X]
    for i in range(m+1):
        Z.append(np.dot(W[i],A[i])+B[i])
        A.append(sigmoid(np.dot(W[i],A[i])+B[i]))
    return Z, A


#onehot encode Y
def one_hot(Y):
    one_hotY = np.zeros([10,Y.size])
    for j in range(Y.size):
        for i in range(10):
            if Y[j] == i:
                one_hotY[i,j] = 1
    return one_hotY

def back_prop(m, Z, A, W, Y):
    Y = one_hot(Y)
    #initialize list of differentials
    dW=[]
    dB=[]
    dA=[2*(A[m+1]-Y)]
    #compute differential of the activations with respect to the cost
    for i in range(m):
        dA.append(np.dot((dA[i]*sigmoid(Z[m-i])*(1-sigmoid(Z[m-i]))).transpose(),W[m-i]).transpose())
    #compute differentials of weights and biases with respect to the cost
    for i in range(m+1):
        dW.append(1/(A[0].shape[1])*np.dot((sigmoid(Z[m-i])*(1-sigmoid(Z[m-i]))*dA[i]),A[m-i].transpose()))
        dB.append(1/(A[0].shape[1])*np.diag(np.dot(sigmoid(Z[m-i])*(1-sigmoid(Z[m-i])),(dA[i]).transpose())))
    dW.reverse()
    dB.reverse()
    return dW, dB


#update parameters
def update_params(m, W, B, dW, dB, alpha):
    for i in range(m+1):
        W[i] = W[i]-alpha*dW[i]
        B[i] = B[i]-alpha*dB[i][:,np.newaxis]
    return W, B


#get the prediction
def get_predictions(m, A):
    return np.argmax(A[m+1],0)


#get accuracy
def get_accuracy(predictions, Y):
    m = 0
    T = Y.size
    for i in range(T):
        if Y[i] == predictions[i]:
            m=m+1
    return m/(T)

#gradient descent
def gradient_descent(m, X, Y, iterations, alpha, W, B):
    for i in range(iterations):
        Z, A = forward_prop(m, W, B, X)
        dW, dB = back_prop(m, Z, A, W, Y)
        W, B = update_params(m, W, B, dW, dB, alpha)
        #if i % 10 == 0:
        #    a = get_accuracy(get_predictions(A_2),Y)
        #    print(f"Iterations: {i}. Accuracy: {a*100}%.")
    return W, B, A

#def stochastic_gradient_descent(n, m, X_trainsamples, Y_trainsamples, iterations, alpha):
#    W, B, m = init(n,m)
#    i=0
#    while i == 0 or get_accuracy(get_predictions(m,A), Y_trainsamples[59]) <= 0.97:
#        for j in range(len(X_trainsamples)):
 #           W, B, A = gradient_descent(m, X_trainsamples[j],Y_trainsamples[j], 1, alpha, W, B)
 #           if i % 10 == 0 and j == 0:
 #               a = get_accuracy(get_predictions(m, A),Y_trainsamples[j])
 #               print(f"Iterations: {i}. Accuracy: {a*100}%.")
 #       i = i+1
 #   print("The network has finished training.")
 #   return W, B
#stochastic gradient descent
def stochastic_gradient_descent(n, m, X_trainsamples, Y_trainsamples, iterations, alpha):
    W, B, m = init(n,m)
    for i in range(iterations):
        for j in range(len(X_trainsamples)):
            W, B, A = gradient_descent(m, X_trainsamples[j],Y_trainsamples[j], 1, alpha, W, B)
            if i % 10 == 0 and j == 0:
                a = get_accuracy(get_predictions(m, A),Y_trainsamples[j])
                print(f"Iterations: {i}. Accuracy: {a*100}%.")
        if get_accuracy(get_predictions(m,A), Y_trainsamples[59]) >= 0.92:
            print("The network has finished training.")
            break
    return W, B


#save model
W, B = stochastic_gradient_descent(10, 2, X_trainsamples,Y_trainsamples,50000,1)
for i in range(len(W)):
    np.savetxt(f"W_{i}.csv",W[i],delimiter=",")
for i in range(len(B)):
    np.savetxt(f"B_{i}.csv",B[i],delimiter=",")


#test the model
def test_model(m, W, B, X, Y):
    Z, A = forward_prop(m, W, B, X)
    print(f"Accuracy on test data: {get_accuracy(get_predictions(m, A), Y)*100}%")



test_model(2, W, B, X_test, Y_test)
