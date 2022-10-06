import numpy as np
import pandas as pd


#Load the data
df_train = pd.read_csv(r"C:\Users\Philippe\Documents\python codes\neural network hand written digits\mnist_train.csv")
df_test = pd.read_csv(r"C:\Users\Philippe\Documents\python codes\neural network hand written digits\mnist_test.csv")
df_train = pd.DataFrame(df_train).to_numpy().transpose()
df_test = pd.DataFrame(df_test).to_numpy().transpose()
print(f"The training data is a 2 dimensional array of following shape: {df_train.shape}")
print(f"The testing data is a 2 dimensional array of following shape: {df_test.shape}")
X_test = df_test[1:,:]
X_train = df_train[1:,:]/255
Y_test = df_test[0,:]
Y_train = df_train[0,:]

x = np.array([[1,1,2],[1,1,1]])+np.array([[-1],[3]])
y = np.array([2,1,2])
#W_1 = np.random.randn(10,784)
#c=np.dot(W_1,X_train)
#print(c.shape)
Y=np.array([0,2,1,3,3])

#print(type(Y[1]))


#initialize the weights ans biases of the network
def init():
    W_1 = np.random.rand(10,784)-0.5
    W_2 = np.random.rand(10,10)-0.5
    B_1 = np.random.rand(10,1)-0.5
    B_2 = np.random.rand(10,1)-0.5
    return W_1, W_2, B_1, B_2


#define Sigmoid function
def sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)



#forward propagation
def forward_prop(W_1, W_2, B_1, B_2, X):
    Z_1 = np.dot(W_1,X)+B_1
    A_1 = sigmoid(Z_1)
    Z_2 = np.dot(W_2, A_1)+B_2
    A_2 = sigmoid(Z_2)
    return Z_1, Z_2, A_1, A_2


#onehot encode Y
def one_hot(Y):
    one_hotY = np.zeros([10,Y.size])
    for j in range(Y.size):
        for i in range(10):
            if Y[j] == i:
                one_hotY[i,j] = 1
    return one_hotY


#back propagation
def back_prop(Z_1, Z_2, A_1, A_2, W_2, Y, X):
    Y = one_hot(Y)
    dW_2 = 1/59999*2*np.dot((A_2-Y),(sigmoid(Z_2)*(1-sigmoid(Z_2))*A_1).transpose())
    predB_2 = 1/59999*np.diag(np.dot(sigmoid(Z_2)*(1-sigmoid(Z_2)),(2*(A_2-Y)).transpose()))
    dB_2 = np.zeros([10,1])
    for i in range(10):
        dB_2[i,0] = predB_2[i]
    dA_1 = 2*np.dot((A_2-Y).transpose(),W_2).transpose()
    dW_1 = 1/59999*np.dot(dA_1*(sigmoid(Z_1)*(1-sigmoid(Z_1))),X.transpose())
    predB_1 = 1/59999*np.diag(np.dot(sigmoid(Z_1)*(1-sigmoid(Z_1)),dA_1.transpose()))
    dB_1 = np.zeros([10,1])
    for i in range(10):
        dB_1[i,0] = predB_1[i]
    return dW_1, dB_1, dW_2, dB_2


#update parameters
def update_params(W_1, B_1, W_2, B_2, dW_1, dB_1, dW_2, dB_2, alpha):
    W_1 = W_1-alpha*dW_1
    W_2 = W_2-alpha*dW_2
    B_1 = B_1-alpha*dB_1
    B_2 = B_2-alpha*dB_2
    return W_1, W_2, B_1, B_2

#get the prediction
def get_predictions(A_2):
    return np.argmax(A_2,0)

#get accuracy
def get_accuracy(predictions, Y):
    m = 0
    T = Y.size
    for i in range(T):
        if Y[i] == predictions[i]:
            m=m+1
    return m/(T)

#gradient descent
def gradient_descent(X, Y, iterations, alpha):
    W_1, W_2, B_1, B_2 = init()
    for i in range(iterations):
        Z_1, Z_2, A_1, A_2 = forward_prop(W_1, W_2, B_1, B_2, X)
        dW_1, dB_1, dW_2, dB_2 = back_prop(Z_1, Z_2, A_1, A_2, W_2, Y, X)
        W_1, W_2, B_1, B_2 = update_params(W_1, B_1, W_2, B_2, dW_1, dB_1, dW_2, dB_2, alpha)
        if i % 10 == 0:
            a = get_accuracy(get_predictions(A_2),Y)
            print(f"Iterations: {i}. Accuracy: {a*100}%.")
    return W_1, W_2, B_1, B_2
gradient_descent(X_train, Y_train, 100000000000, 1)

