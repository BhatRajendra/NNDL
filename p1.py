import numpy as np
import random
import math
eta = 0.1
inp = np.array([[0,0],[0,1],[1,0],[1,1]])
target = np.array([0,0,0,1])
w=np.array([0.0,0.0])

for i in range(len(w)):
    w[i]=random.uniform(0, 1)
print(w)

bias = 0.2

def sigmoid(x):
    return 1/(1+np.exp(-x))


def weightupdate(inpu, we, output):
    t = inpu[0] & inpu[1]
    for i in range(len(we)):
        delta = eta*(target-output)*inpu[i]
        we[i] +=  delta

def forward(inputs):
    sum= 0
    sum+=bias
    for i in range(len(inputs)): 
        sum+=(inputs[i]*w[i])
    out = sigmoid(sum)
    print("here",out)
    
    weightupdate(inputs,w,out)
    print("after update")
    print(w)
#main
for i in range(len(inp)):
    forward(inp[i])
    
