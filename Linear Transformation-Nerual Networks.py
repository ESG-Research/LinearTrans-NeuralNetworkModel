
# Linear Transformations and Neural Networks

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils





parameters = utils.initialize_parameters(2)
print(parameters)


# Processce FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m), where n_x is the dimension input (in our example is 2) and m is the number of training samples
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    Y_hat -- The output of size (1, m)
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Implement Forward Propagation to calculate Z.
    
    Z = W @ X + b
    Y_hat = Z

    return Y_hat
    

# The Cost Function used to traing this model is 

def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost


# Model FUNCTION: nn_model

def nn_model(X, Y, num_iterations=1000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (1, number of examples)
    num_iterations -- number of iterations in the loop
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    n_x = X.shape[0]
    
    # Initialize parameters
    parameters = utils.initialize_parameters(n_x) 
    
    # Loop
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "Y_hat".
        Y_hat = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = compute_cost(Y_hat, Y)
        
        # Parameters update.
        parameters = utils.train_nn(parameters, Y_hat, X, Y, learning_rate = 0.001) 
        
        # Print the cost every iteration.
        if print_cost:
            if i%100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

    return parameters



###Application:

# Load a dataset to train the neural network.

df = pd.read_csv("data/toy_dataset.csv")
df.head()


# First turn the data into a numpy array that we can pass to our function.
X = np.array(df[['x1','x2']]).T
Y = np.array(df['y']).reshape(1,-1)


# Run the next block to update the parameters dictionary with the fitted weights.
parameters = nn_model(X,Y, num_iterations = 5000, print_cost= True)


# ## 4 - Make your predictions!
# 
# Now that you have the fitted parameters, you are able to predict any value with your neural network! You just need to perform the following computation:
# 
# $$ Z = W X + b$$ 
# 
# Where $W$ and $b$ are in the parameters dictionary.
# 
# <a name="ex07"></a>
# ### Exercise 7
# 
# Now you will make the predictor function. It will input a parameters dictionary, a set of points X and output the set of predicted values. 

# In[104]:


# GRADED FUNCTION: predict

def predict(X, parameters):

    W = parameters['W']
    b = parameters['b']

    Z = np.dot(W, X) + b

    return Z


# In[105]:


y_hat = predict(X,parameters)


# In[106]:


df['y_hat'] = y_hat[0]


# Now let's check some predicted values versus the original ones:

# In[107]:


for i in range(10):
    print(f"(x1,x2) = ({df.loc[i,'x1']:.2f}, {df.loc[i,'x2']:.2f}): Actual value: {df.loc[i,'y']:.2f}. Predicted value: {df.loc[i,'y_hat']:.2f}")


# Pretty good, right? Congratulations! You have finished the W3 assignment!
