#!/usr/bin/env python
# coding: utf-8

# # The Perceptron algorithm, Part 2
# 
# In this part 2 of the assignment, you will implement the following:
# 
# 1. The Kernel Perceptron training and evaluation procedures.
# 
# 2. Implement the quadratic kernel.
# 
# 3. Show empirically that the Kernel Perceptron converges on the XOR dataset.
# 
# 4. Bonus points.

# ## Write Your Name Here:
# 

# # <font color="blue"> Submission Instructions</font>
# 
# 1. Click the Save button at the top of the Jupyter Notebook.
# 2. Please make sure to have entered your name above.
# 3. Select Cell -> All Output -> Clear. This will clear all the outputs from all cells (but will keep the content of ll cells). 
# 4. Select Cell -> Run All. This will run all the cells in order, and will take several minutes.
# 5. Once you've rerun everything, select File -> Download as -> PDF via LaTeX and download a PDF version *perceptron.pdf* showing the code and the output of all cells, and save it in the same folder that contains the notebook file *perceptron.ipynb*.
# 6. Look at the PDF file and make sure all your solutions are there, displayed correctly. The PDF is the only thing we will see when grading!
# 7. Submit **both** your PDF and notebook on Canvas.

# In[ ]:


import numpy as np
import utils

np.random.seed(1)


# ## 1. The Perceptron algorithm (from Part 1)
# 
# Copy here from Part 1 the implementation of the training procedure for the Perceptron algorithm.
# 
# The training algorithm runs for the specified number of epochs or until convergence, whichever happens first. The algorithm converged when it makes no mistake on the training examples. If the algorithm converged, display a message "Converged in \<e\> epochs!'.

# In[ ]:


def perceptron_train(X, y, E):
    """Perceptron training function.
    Args:
        X (np.ndarray): A 2D array training instances, one per row.
        y (np.ndarray): A vector of labels.
        E (int): the maximum number of epochs.
        
    Returns:
        np.ndarray: The learned vector of weights / parameters.
    """
    # Add bias feature to X.
    X = # YOUR CODE HERE
    
    # Initialize w with zero's.
    w = # YOUR CODE HERE
    
    for e in range(E):
        # YOUR CODE HERE
        
        
        
        
        
    return w


# Implement the Perceptron prediction function that takes as input the Perceptron parameters w and returns a vector with the labels (+1 or -1) predicted for the test examples in input data.

# In[ ]:


def perceptron_test(X, w):
    """Perceptron prediction function.
    Args:
        X (np.ndarray): A 2D array training instances, one per row.
        w (np.ndarray): A vector of parameters.

        
    Returns:
        np.ndarray: The vector of predicted labels.
    """
    # Add bias feature to X.
    X = # YOUR CODE HERE
    
    # Compute the perceptron predictions on the examples in X. 
    pred = # YOUR CODE HERE
    
    
        
    return pred


# ## 2. The Kernel Perceptron
# 
# Implement the training procedure for the Kernel Perceptron algorithm.
# 
# The algorithm runs for the specified number of epochs or until convergence, whichever happens first. The algorithm converged when it makes no mistake on the training examples. If the algorithm converged, display a message "Converged in \<e\> epochs!'.
# 
# Return the learned vector of dual parameters alpha.

# In[ ]:


def kperceptron_train(X, y, E, K):
    """Kernel Perceptron training function.
    Args:
        X (np.ndarray): A 2D array training instances, one per row.
        y (np.ndarray): A vector of labels.
        E (int): the maximum number of epochs.
        K (function): the kernel function.
        
    Returns:
        np.ndarray: The learned vector of dual parameters.
    """
    
    # Initialize dual parameters alpha with zero's.
    alpha = # YOUR CODE HERE
    
    for e in range(E):
        # YOUR CODE HERE
        
        
        
        
        
    return alpha


# Write a function that takes as input the set of training examples, their labels, and the trained dual parameters, and return a tuple containing:
# 
# 1. the support vectors;
# 
# 2. their labels;
# 
# 3. their dual parameters. 
# 
# Remember, support vectors are training examples for which the corresponding dual parameter alpha is non-zero.

# In[ ]:


def support_vectors(X, y, alpha):
    """Select support vectors.
    Args:
        X (np.ndarray): A 2D array training instances, one per row.
        y (np.ndarray): A vector of labels.
        alpha (np.ndarray): The vector of dual parameters.
        
    Returns:
        X_s, y_s, alpha_s: The support vectors, their labels, and their dual parameters.
    """
    
    # YOUR CODE HERE
    X_s = 
    
    y_s = 
    
    alpha_s = 
    
    
    return X_s, y_s, alpha_s


# Implement the Kernel Perceptron prediction function that takes as input the Kernel Perceptron dual parameters alpha, support vectors, their labels, and the kernel, and returns a vector with the labels (+1 or -1) predicted for the test examples.

# In[ ]:


def kperceptron_test(X_s, y_s, alpha_s, K, X):
    """Kernel Perceptron prediction function.
    Args:
        X_s (np.ndarray): A 2D array of support vectors, one per row.
        y_s (np.ndarray): The vector of labels corresponding to the support vectors.
        alpha_S (np.ndarray): The vector of dual parameters for the support vectors.
        K (function): the kernel function.

        X (np.ndarray): A 2D array of test instances, one per row.
        
    Returns:
        np.ndarray: The vector of predicted labels for the test instances.
    """
    
    # Compute the kernel perceptron predictions on the examples in X. 
    # YOUR CODE HERE
    pred = 
    
    
        
    return pred


# Here is my implementation of the linear kernel $K(x,y) = 1 + x^T y$

# In[ ]:


def linear_kernel(x, y):
    return 1 + x @ y


# Implement the quadratic kernel $K(x, y) = (1 + x^T y)^2$.

# In[ ]:


def quadratic_kernel(x, y):
    # YOUR CODE HERE
    result = 
    
    return result


# ## 3.  Let's experiment with the XOR dataset and the two perceptron algorithms.

# In[ ]:


# Create a more complex XOR dataset, with N points clustered around each of the 4 'corners'
def xor_dataset(N):
    """Generate XOR dataset.
    Args:
        N: number of points per example cluster.
        
    Returns:
        X: A 2D array with examples, one per line.
        y: A vector of labels.
    """
    X00 = (0, 0) + (np.random.sample((N, 2)) - 0.5) / 4
    y00 = np.full((N,), -1)
    X01 = (0, 1) + (np.random.sample((N, 2)) - 0.5) / 4
    y01 = np.full((N,), +1)
    X10 = (1, 0) + (np.random.sample((N, 2)) - 0.5) / 4
    y10 = np.full((N,), +1)
    X11 = (1, 1) + (np.random.sample((N, 2)) - 0.5) / 4
    y11 = np.full((N,), -1)
    
    X = np.row_stack((X00, X01, X10, X11))
    y = np.concatenate((y00, y01, y10, y11))
    
    return X, y


# In[ ]:


X, y = xor_dataset(5)

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

plt.scatter(X[:,0], X[:,1], c = y)


# Let's train the Kernel Perceptron algorithm with the quadratic kernel for up to 50 epochs and evaluate its accuracy on a new set of test examples.

# In[ ]:


# Train the kernel perceptron
E = 50
alpha = kperceptron_train(X, y, E, quadratic_kernel)
print('alpha =', alpha)

# Find the support vectors.
X_s, y_s, alpha_s = support_vectors(X, y, alpha)
print('Found', alpha_s.size, 'support vectors out of', alpha.size, 'training examples.')

# Test the kernel perceptron on a new set of test examples.
X_test, y_test = xor_dataset(2)
predictions = kperceptron_test(X_s, y_s, alpha_s, quadratic_kernel, X_test)


# Write code to compute the accuracy of the Kernel Perceptron on the test examples.

# In[ ]:


# YOUR CODE HERE


accuracy = 
print('Accuracy on test examples is:', accuracy)


# ## 4. Bonus points
# 
# A. Here, train the Perceptron algorithm on the XOR dataset, using the feature space corresponding to the quadratic kernel.
# 
# B. Empirically show that the trained weight vector is equal with the weighted sum of the support vectors (slide 39) that were trained by the Kernel Perceptron on the same dataset.

# In[ ]:


def project_features(X):
    # Project the examples in X, which contain only 2 features x and y, into the feature space
    #     corresponding to the quadratic kernel shown on slide 45, i.e. from 2 to 6 features.
    # Return the array of the new examples, each containing 6 features.
    


# In[ ]:


# Train the perceptron on the projected version of the XOR dataset, using the 6 features.
Xnew = project_features(X)
E = 50
w = train(Xnew, y, E)
print(w)


# In[ ]:


# Compute the weighted sum of the support vectors (slide 39) using X_s, y_s, alpha_s.
kw = # YOUR CODE HERE


# Compare with w printed above. They should be the same.
print(kw)


# ## 5. Anything extra goes here.

# In[ ]:




