import numpy as np

a = np.array([[1,2,2],[1,1,1]])
print(a)
print(a.shape[0])
print(type(a))
W_2 = np.random.rand(10,10)-0.5
print(W_2)
print(W_2.shape)
B = []
for i in range(1):                     #weights for hidden layers
    B.append(np.zeros([10,1]))
B.append(np.zeros([10,1]))
print(B)
print(B[0].shape)