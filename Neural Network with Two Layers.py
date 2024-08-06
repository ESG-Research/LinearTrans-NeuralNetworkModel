
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
# A function to create a dataset.
from sklearn.datasets import make_blobs

# Output of plotting commands is displayed inline within the Jupyter notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

# Set a seed so that the results are consistent.
np.random.seed(3)


#Dataset
#我们有一个观测点为2000的dataset
m = 2000
samples, labels = make_blobs(n_samples=m, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0) | (labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 0
X = np.transpose(samples)
Y = labels.reshape((1, m))

plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

print ('The shape of X is: ' + str(X.shape))
print ('The shape of Y is: ' + str(Y.shape))
print ('I have m = %d training examples!' % (m))

#因为要建立一个Classification 神经网络模型，因此要先设置Active Function，西格玛(Z), Sigmoid(Z)=1/1-e^-z
#Define Activation Function
def sigmoid(z):
    ### START CODE HERE ### (~ 1 line of code)
    res = 1/(1+np.exp(-z))
    ### END CODE HERE ###
    
    return res


#在设置完Active Function之后，我们可以开始建立NN model，这里我们建立一个两层的NN Model：
#Implementation of the Neural Network Model with Two Layers

#首先要确定我们的神经网络模型的结构(几层几级)
# Defining the Neural Network Structure
########
########**********必须注意，Nureal Network中所说的Layer(层)指的是数据或值的Array，不是指感受器**********##########
######## (data value)输入层————感受器1————数值1(Hidden层)————感受器2————(Y_hat value)输出层############

# Define three variables:
# - `n_x`: the size of the input layer 输入层 也就是dataset_array
# - `n_h`: the size of the hidden layer (set it equal to 2 for 2层神经网络模型）
# - `n_y`: the size of the output layer 输出层 也就是Y_hat array

# 这里我们使用 shapes of X and Y to find n_x and n_y:
# the size of the input layer n_x equals to the size of the input vectors placed in the columns of the array X
# the outpus for each of the data point will be saved in the columns of the the array Y：

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
  
    # Size of input layer.
    n_x = X.shape[0]
    # Size of hidden layer.
    n_h = 2
    # Size of output layer.
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

(n_x, n_h, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

# ##### __Expected Output__
# 
# ```Python
# The size of the input layer is: n_x = 2
# The size of the hidden layer is: n_h = 2
# The size of the output layer is: n_y = 1
# ```


# 构建好我们的神经网络模型结构之后，可以开始设置初始化参数。
# Implement the function `initialize_parameters()`.
# - You will initialize the weights matrix with random values. 
#     - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
# - You will initialize the bias vector as zeros. 
#     - Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.
# 这里的初始化，主要是初始化 W， b，而且重点是给定这两者Shape，因为X和Y的shape是来源于dataset是固定的，我们的w，b是任意形状，因此要去和X，Y形状匹配
# 匹配是指运算 W * X，则 W（n，nx）和 X（Xn x m)对应， b （1 x m）和 Y（1 x m）对应
# 此模型：X1,X2 因此输入层X 是 2 row m column。Hidden Layers因为是2个感受器那就是 2 row。Y_hat是一个输出值，因此是 1 row m columns。
######      n_x = 2 , n_h = 2, n_y = 1      ######
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


parameters = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# ##### __Expected Output__ 
# ```Python
# W1 = [[ 0.01788628  0.0043651 ]
#  [ 0.00096497 -0.01863493]]
# b1 = [[0.]
#  [0.]]
# W2 = [[-0.00277388 -0.00354759]]
# b2 = [[0.]]
# ```



当我们设置好Active Function，Model Structure，以及W，b的shape之后，我们可以开始设置Loop 循环以实现算法：


#The Loop

# Implement `forward_propagation()`.
# - The steps you have to implement are:
#     1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
#     2. Implement Forward Propagation. Compute `Z1` multiplying matrices `W1`, `X` and adding vector `b1`. 
#     Then find `A1` using the `sigmoid` activation function. Perform similar computations for `Z2` and `A2`.


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- the sigmoid output of the second activation
    cache -- python dictionary containing Z1, A1, Z2, A2 
    (that simplifies the calculations in the back propagation step)
    """
    # Retrieve each parameter from the dictionary "parameters".

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    
    # Implement forward propagation to calculate A2.

    Z1 = np.matmul(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = sigmoid(Z2)
    ### END CODE HERE ###
    
    assert(A2.shape == (n_y, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

A2, cache = forward_propagation(X, parameters)

print(A2)


# Define a cost function “LOGLOSS”
# LOGLOSS = L(w,b) = 1/m*SUM(-y*lna - (1-y)ln(1-a))


def compute_cost(A2, Y):
    """
    Computes the cost function as a log loss
    
    Arguments:
    A2 -- The output of the neural network of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    cost -- log loss
    
    """
    # Number of examples.
    m = Y.shape[1]
    
    ### START CODE HERE ### (~ 2 lines of code)
    logloss = -np.multiply(np.log(A2),Y) - np.multiply(np.log(1-A2),1-Y)
    cost = 1/m*np.sum(logloss)
    ### END CODE HERE ###

    assert(isinstance(cost, float))
    
    return cost


# In[97]:


print("cost = " + str(compute_cost(A2, Y)))


# ##### __Expected Output__ 
# Note: the elements of the arrays W1 and W2 maybe be different!
# 
# ```Python
# cost = 0.6931477703826823
# ```

# In[98]:


# Note: 
# Actual values are not checked here in the unit tests (due to random initialization).
w3_unittest.test_compute_cost(compute_cost, A2)


# Calculate partial derivatives as shown in $(15)$:
# 
# \begin{align}
# \frac{\partial \mathcal{L} }{ \partial W^{[2]} } &= 
# \frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\\
# \frac{\partial \mathcal{L} }{ \partial b^{[2]} } &= 
# \frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1},\\
# \frac{\partial \mathcal{L} }{ \partial W^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T,\\
# \frac{\partial \mathcal{L} }{ \partial b^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1}.\\
# \end{align}

# In[99]:


def backward_propagation(parameters, cache, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- python dictionary containing Z1, A1, Z2, A2
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate partial derivatives denoted as dW1, db1, dW2, db2 for simplicity. 
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

grads = backward_propagation(parameters, cache, X, Y)

print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))


# <a name='ex06'></a>
# ### Exercise 6
# 
# Implement `update_parameters()`.
# 
# **Instructions**:
# - Update parameters as shown in $(9)$ (section [2.3](#2.3)):
# \begin{align}
# W^{[1]} &= W^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[1]} },\\
# b^{[1]} &= b^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[1]} },\\
# W^{[2]} &= W^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[2]} },\\
# b^{[2]} &= b^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[2]} }.\\
# \end{align}
# - The steps you have to implement are:
#     1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
#     2. Retrieve each derivative from the dictionary "grads" (which is the output of `backward_propagation()`) by using `grads[".."]`.
#     3. Update parameters.

# In[100]:


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients
    learning_rate -- learning rate for gradient descent
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters".
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Retrieve each gradient from the dictionary "grads".
    ### START CODE HERE ### (~ 4 lines of code)
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    ### END CODE HERE ###
    
    # Update rule for each parameter.
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    ### END CODE HERE ###
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[101]:


parameters_updated = update_parameters(parameters, grads)

print("W1 updated = " + str(parameters_updated["W1"]))
print("b1 updated = " + str(parameters_updated["b1"]))
print("W2 updated = " + str(parameters_updated["W2"]))
print("b2 updated = " + str(parameters_updated["b2"]))


# ##### __Expected Output__ 
# Note: the actual values can be different!
# 
# ```Python
# W1 updated = [[ 0.01790427  0.00434496]
#  [ 0.00099046 -0.01866419]]
# b1 updated = [[-6.13449205e-07]
#  [-8.47483463e-07]]
# W2 updated = [[-0.00238219 -0.00323487]]
# b2 updated = [[0.00094478]]
# ```

# In[102]:


w3_unittest.test_update_parameters(update_parameters)


# <a name='3.4'></a>
# ### 3.4 - Integrate parts 3.1, 3.2 and 3.3 in nn_model()

# <a name='ex07'></a>
# ### Exercise 7
# 
# Build your neural network model in `nn_model()`.
# 
# **Instructions**: The neural network model has to use the previous functions in the right order.

# In[121]:


# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    learning_rate -- learning rate parameter for gradient descent
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters.
    ### START CODE HERE ### (~ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###
    
    # Loop.
    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (~ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y". Outputs: "cost".
        cost = compute_cost(A2, Y)
        
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        
        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


# In[122]:


parameters = nn_model(X, Y, n_h=2, num_iterations=3000, learning_rate=1.2, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]


# ##### __Expected Output__ 
# Note: the actual values can be different!
# 
# ```Python
# Cost after iteration 0: 0.693148
# Cost after iteration 1: 0.693147
# Cost after iteration 2: 0.693147
# Cost after iteration 3: 0.693147
# Cost after iteration 4: 0.693147
# Cost after iteration 5: 0.693147
# ...
# Cost after iteration 2995: 0.209524
# Cost after iteration 2996: 0.208025
# Cost after iteration 2997: 0.210427
# Cost after iteration 2998: 0.208929
# Cost after iteration 2999: 0.211306
# W1 = [[ 2.14274251 -1.93155541]
#  [ 2.20268789 -2.1131799 ]]
# b1 = [[-4.83079243]
#  [ 6.2845223 ]]
# W2 = [[-7.21370685  7.0898022 ]]
# b2 = [[-3.48755239]]
# ```

# In[123]:


# Note: 
# Actual values are not checked here in the unit tests (due to random initialization).
w3_unittest.test_nn_model(nn_model)


# The final model parameters can be used to find the boundary line and for making predictions. 

# <a name='ex08'></a>
# ### Exercise 8
# 
# Computes probabilities using forward propagation, and make classification to 0/1 using 0.5 as the threshold.

# In[106]:


# GRADED FUNCTION: predict

def predict(X, parameters):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (blue: 0 / red: 1)
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    ### END CODE HERE ###
    
    return predictions


# In[107]:


X_pred = np.array([[2, 8, 2, 8], [2, 8, 8, 2]])
Y_pred = predict(X_pred, parameters)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")


# ##### __Expected Output__ 
# 
# ```Python
# Coordinates (in the columns):
# [[2 8 2 8]
#  [2 8 8 2]]
# Predictions:
# [[ True  True False False]]
# ```

# In[108]:


w3_unittest.test_predict(predict)


# Let's visualize the boundary line. Do not worry if you don't understand the function `plot_decision_boundary` line by line - it simply makes prediction for some points on the plane and plots them as a contour plot (just two colors - blue and red).

# In[109]:


def plot_decision_boundary(predict, parameters, X, Y):
    # Define bounds of the domain.
    min1, max1 = X[0, :].min()-1, X[0, :].max()+1
    min2, max2 = X[1, :].min()-1, X[1, :].max()+1
    # Define the x and y scale.
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # Create all of the lines and rows of the grid.
    xx, yy = np.meshgrid(x1grid, x2grid)
    # Flatten each grid to a vector.
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((1, len(r1))), r2.reshape((1, len(r2)))
    # Vertical stack vectors to create x1,x2 input for the model.
    grid = np.vstack((r1,r2))
    # Make predictions for the grid.
    predictions = predict(grid, parameters)
    # Reshape the predictions back into a grid.
    zz = predictions.reshape(xx.shape)
    # Plot the grid of x, y and z values as a surface.
    plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral.reversed())
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

# Plot the decision boundary.
plot_decision_boundary(predict, parameters, X, Y)
plt.title("Decision Boundary for hidden layer size " + str(n_h))

# grade-up-to-here 


# That's great, you can see that more complicated classification problems can be solved with two layer neural network!

# <a name='4'></a>
# ## 4 - Optional: Other Dataset

# Build a slightly different dataset:

# In[110]:


n_samples = 2000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0)] = 0
labels[(labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 1
X_2 = np.transpose(samples)
Y_2 = labels.reshape((1,n_samples))

plt.scatter(X_2[0, :], X_2[1, :], c=Y_2, cmap=colors.ListedColormap(['blue', 'red']));


# Notice that when building your neural network, a number of the nodes in the hidden layer could be taken as a parameter. Try to change this parameter and investigate the results:

# In[119]:


# parameters_2 = nn_model(X_2, Y_2, n_h=1, num_iterations=3000, learning_rate=1.2, print_cost=False)
parameters_2 = nn_model(X_2, Y_2, n_h=3, num_iterations=3000, learning_rate=1.2, print_cost=False)
# parameters_2 = nn_model(X_2, Y_2, n_h=15, num_iterations=3000, learning_rate=1.2, print_cost=False)

# This function will call predict function 
plot_decision_boundary(predict, parameters_2, X_2, Y_2)
plt.title("Decision Boundary")


# You can see that there are some misclassified points - real-world datasets are usually linearly inseparable, and there will be a small percentage of errors. More than that, you do not want to build a model that fits too closely, almost exactly to a particular set of data - it may fail to predict future observations. This problem is known as **overfitting**.

# Congrats on finishing this programming assignment!

# In[ ]:




