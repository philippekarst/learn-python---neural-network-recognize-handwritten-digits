from tkinter.messagebox import NO
import pandas as pd
import numpy as np

#load the model
W_1 = pd.read_csv(r".\W_1.csv", header=None)
W_2 = pd.read_csv(r".\W_2.csv", header=None)
B_1 = pd.read_csv(r".\B_1.csv", header=None)
B_2 = pd.read_csv(r".\B_2.csv", header=None)
W_1 = pd.DataFrame(W_1).to_numpy()
W_2 = pd.DataFrame(W_2).to_numpy()
B_1 = np.squeeze(pd.DataFrame(B_1).to_numpy())
B_2 = np.squeeze(pd.DataFrame(B_2).to_numpy())

#load the a picture that must be recognized
pic = pd.read_csv(r".\mnist_test.csv")
pic = pd.DataFrame(pic).to_numpy().transpose()
picx = pic
label = pic[0,8]
pics = pic[1:,:]/255


#define Sigmoid function
def sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)

#run it through the network
def forward_prop(X):
    Z_1 = np.dot(W_1,X)+B_1
    A_1 = sigmoid(Z_1)
    Z_2 = np.dot(W_2, A_1)+B_2
    A_2 = sigmoid(Z_2)
    return A_2

def get_predictions(A_2):
    return np.argmax(A_2,0)



for i in range(10):
    label = picx[0,i]
    pic = pics[:,i]
    A_2 = forward_prop(pic)
    print(f"Network prediction: {get_predictions(A_2)}")
    print(f"Label:{label}")

