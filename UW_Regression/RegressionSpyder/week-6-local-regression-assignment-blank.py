
# coding: utf-8

# # Predicting house prices using k-nearest neighbors regression
# In this notebook, you will implement k-nearest neighbors regression. You will:
#   * Find the k-nearest neighbors of a given query input
#   * Predict the output for the query input using the k-nearest neighbors
#   * Choose the best value of k using a validation set

# # Fire up GraphLab Create

# In[ ]:

import graphlab


# # Load in house sales data

# For this notebook, we use a subset of the King County housing dataset created by randomly selecting 40% of the houses in the full dataset.

# In[ ]:

sales = graphlab.SFrame('kc_house_data_small.gl/')


# # Import useful functions from previous notebooks

# To efficiently compute pairwise distances among data points, we will convert the SFrame into a 2D Numpy array.
# First import the numpy library and then copy and paste `get_numpy_data()` from the second notebook of Week 2.

# In[ ]:

import numpy as np # note this allows us to refer to numpy as np instead


# In[ ]:

def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]    
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)


# We will also need the `normalize_features()` function from Week 5 that normalizes all feature columns to unit norm.
# Paste this function below.

# In[ ]:

def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return (normalized_features, norms)

# # Split data into training, test, and validation sets

# In[ ]:

(train_and_validation, test) = sales.random_split(.8, seed=1) # initial train/test split
(train, validation) = train_and_validation.random_split(.8, seed=1) # split training set into training and validation sets


# # Extract features and normalize

# Using all of the numerical inputs listed in `feature_list`, transform the training, test, and validation SFrames into Numpy arrays:

# In[ ]:

feature_list = ['bedrooms',  
                'bathrooms',  
                'sqft_living',  
                'sqft_lot',  
                'floors',
                'waterfront',  
                'view',  
                'condition',  
                'grade',  
                'sqft_above',  
                'sqft_basement',
                'yr_built',  
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')


# In computing distances, it is crucial to normalize features. Otherwise, for example, the `sqft_living` feature (typically on the order of thousands) would exert a much larger influence on distance than the `bedrooms` feature (typically on the order of ones). We divide each column of the training feature matrix by its 2-norm, so that the transformed column has unit norm.
# 
# IMPORTANT: Make sure to store the norms of the features in the training set. The features in the test and validation sets must be divided by these same norms, so that the training, test, and validation sets are normalized consistently.

# In[ ]:

features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms


# # Compute a single distance

# To start, let's just explore computing the "distance" between two given houses.
# We will take our **query house** to be the first house of the test set and look at the distance
# between this house and the 10th house of the training set.
# 
# To see the features associated with the query house, print the first row (index 0) of the test feature matrix.
# You should get an 18-dimensional vector whose components are between 0 and 1.

# In[ ]:

first_test = features_test[0]
print first_test

# Now print the 10th row (index 9) of the training feature matrix. Again, you get an 18-dimensional vector with components between 0 and 1.

# In[ ]:

tenth_train = features_train[9]
print tenth_train


# ***QUIZ QUESTION ***
# 
# What is the Euclidean distance between the query house and the 10th house of the training set? 
# 
# Note: Do not use the `np.linalg.norm` function; use `np.sqrt`, `np.sum`, and the power operator (`**`) instead.
# The latter approach is more easily adapted to computing multiple distances at once.

# In[ ]:

dist = np.sqrt(np.sum((first_test-tenth_train)**2))
print dist

# np.linalg.norm(first_test-tenth_train)


# # Compute multiple distances

# Of course, to do nearest neighbor regression, we need to compute the distance between our query house 
# and *all* houses in the training set.  
# 
# To visualize this nearest-neighbor search, let's first compute the distance from our 
# query house (`features_test[0]`) to the first 10 houses of the training set (`features_train[0:10]`) and 
# then search for the nearest neighbor within this small set of houses.  Through restricting ourselves to a 
# small set of houses to begin with, we can visually scan the list of 10 distances to verify that our code for 
# finding the nearest neighbor is working.
# 
# Write a loop to compute the Euclidean distance from the query house to each of the first 10 houses in the
# training set.

# In[ ]:

dists = [(i, np.sqrt(np.sum((first_test-features_train[i])**2))) for i in range(10)]
print dists


# *** QUIZ QUESTION ***
# 
# Among the first 10 training houses, which house is the closest to the query house?

# In[ ]:

print 1


# It is computationally inefficient to loop over computing distances to all houses in our training dataset. 
# Fortunately, many of the Numpy functions can be **vectorized**, applying the same operation over multiple 
# values or vectors.  We now walk through this process.

# Consider the following loop that computes the element-wise difference between the features of the query 
# house (`features_test[0]`) and the first 3 training houses (`features_train[0:3]`):

# In[ ]:

for i in xrange(3):
    print features_train[i]-features_test[0]
    # should print 3 vectors of length 18


# The subtraction operator (`-`) in Numpy is vectorized as follows:

# In[ ]:

print features_train[0:3] - features_test[0]


# Note that the output of this vectorized operation is identical to that of the loop above,
# which can be verified below:

# In[ ]:

# verify that vectorization works
results = features_train[0:3] - features_test[0]
print results[0] - (features_train[0]-features_test[0])
# should print all 0's if results[0] == (features_train[0]-features_test[0])
print results[1] - (features_train[1]-features_test[0])
# should print all 0's if results[1] == (features_train[1]-features_test[0])
print results[2] - (features_train[2]-features_test[0])
# should print all 0's if results[2] == (features_train[2]-features_test[0])


# Aside: it is a good idea to write tests like this cell whenever you are vectorizing a complicated operation.

# # Perform 1-nearest neighbor regression
# 
# Now that we have the element-wise differences, it is not too hard to compute the Euclidean distances 
# between our query house and all of the training houses. First, write a single-line expression to define a 
# variable `diff` such that `diff[i]` gives the element-wise difference between the features of the query house 
# and the `i`-th training house.

# In[ ]:

diff = features_train - first_test


# To test the code above, run the following cell, which should output a value -0.0934339605842:

# In[ ]:

print diff[-1].sum() # sum of the feature differences between the query and last training house
# should print -0.0934339605842


# The next step in computing the Euclidean distances is to take these feature-by-feature differences in `diff`,
# square each, and take the sum over feature indices.  That is, compute the sum of square feature differences for 
# each training house (row in `diff`).
# 
# By default, `np.sum` sums up everything in the matrix and returns a single number. To instead sum only over a row
# or column, we need to specifiy the `axis` parameter described in the `np.sum` 
# [documentation](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html).
# In particular, `axis=1` computes the sum across each row.
# 
# Below, we compute this sum of square feature differences for all training houses and verify that the 
# output for the 16th house in the training set is equivalent to having examined only the 16th row of `diff` 
# and computing the sum of squares on that row alone.

# In[ ]:

print np.sum(diff**2, axis=1)[15] # take sum of squares across each row, and print the 16th sum
print np.sum(diff[15]**2) # print the sum of squares for the 16th row -- should be same as above


# With this result in mind, write a single-line expression to compute the Euclidean distances between the query house 
# and all houses in the training set. Assign the result to a variable `distances`.
# 
# **Hint**: Do not forget to take the square root of the sum of squares.

# In[ ]:

distances = np.sqrt(np.sum(diff**2, axis=1))


# To test the code above, run the following cell, which should output a value 0.0237082324496:

# In[ ]:

print distances[100] # Euclidean distance between the query house and the 101th training house
# should print 0.0237082324496


# Now you are ready to write a function that computes the distances from a query house to all training houses.
# The function should take two parameters: (i) the matrix of training features and 
# (ii) the single feature vector associated with the query.

# In[ ]:

def calc_distances(features, query):
    return np.sqrt(np.sum((features-query)**2, axis=1))


# *** QUIZ QUESTIONS ***
# 
# 1.  Take the query house to be third house of the test set (`features_test[2]`).
# What is the index of the house in the training set that is closest to this query house?

# 2.  What is the predicted value of the query house based on 1-nearest neighbor regression?

# In[ ]:

distances = calc_distances(features_train, features_test[2])
idx = np.argmin(distances)
print idx


# In[ ]:

print output_train[idx]


# # Perform k-nearest neighbor regression

# For k-nearest neighbors, we need to find a *set* of k houses in the training set closest to a given query house. We then make predictions based on these k nearest neighbors.

# ## Fetch k-nearest neighbors
# 
# Using the functions above, implement a function that takes in
#  * the value of k;
#  * the feature matrix for the training houses; and
#  * the feature vector of the query house
#  
# and returns the indices of the k closest training houses. For instance, with 2-nearest neighbor,
# a return value of [5, 10] would indicate that the 6th and 11th training houses are closest to the query house.
# 
# **Hint**: Look at the [documentation for `np.argsort`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html).

# In[ ]:

def find_closest_k(k, features, query):
    distances = calc_distances(features, query)
    index_array = np.argsort(distances)
    return index_array[0:k]
    


# *** QUIZ QUESTION ***
# 
# Take the query house to be third house of the test set (`features_test[2]`). 
# What are the indices of the 4 training houses closest to the query house?

# In[ ]:

idxs = find_closest_k(4, features_train, features_test[2])
print idxs


# ## Make a single prediction by averaging k nearest neighbor outputs

# Now that we know how to find the k-nearest neighbors, write a function that predicts the value of a 
# given query house. **For simplicity, take the average of the prices of the k nearest neighbors in 
# the training set**. The function should have the following parameters:

#  * the value of k;
#  * the feature matrix for the training houses;
#  * the output values (prices) of the training houses; and
#  * the feature vector of the query house, whose price we are predicting.
#  
# The function should return a predicted value of the query house.
# 
# **Hint**: You can extract multiple items from a Numpy array using a list of indices. For instance, `output_train[[6, 10]]` returns the prices of the 7th and 11th training houses.

# In[ ]:

def k_nn_prediction(k, features, output, query):
    idxs = find_closest_k(k, features, query)
    outputs = output[idxs]
    return np.average(outputs)


# *** QUIZ QUESTION ***
# 
# Again taking the query house to be third house of the test set (`features_test[2]`),
# predict the value of the query house using k-nearest neighbors with `k=4` and the simple averaging 
# method described and implemented above.

# In[ ]:

prediction = k_nn_prediction(4, features_train, output_train, features_test[2])
print prediction


# Compare this predicted value using 4-nearest neighbors to the predicted value using 1-nearest neighbor computed earlier.

# ## Make multiple predictions

# Write a function to predict the value of *each and every* house in a query set. (The query set can be any subset of the dataset, be it the test set or validation set.) The idea is to have a loop where we take each house in the query set as the query house and make a prediction for that specific house. The new function should take the following parameters:
#  * the value of k;
#  * the feature matrix for the training houses;
#  * the output values (prices) of the training houses; and
#  * the feature matrix for the query set.
#  
# The function should return a set of predicted values, one for each house in the query set.
# 
# **Hint**: To get the number of houses in the query set, use the `.shape` field of the query features matrix. See [the documentation](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ndarray.shape.html).

# In[ ]:

def k_nn_predictions(k, features, output, queries):
    n = np.shape(queries)[0]
    result = np.zeros(n)
    for i in range(n):
        result[i] = k_nn_prediction(k, features, output, queries[i])
    return result


# *** QUIZ QUESTION ***
# 
# Make predictions for the first 10 houses in the test set using k-nearest neighbors with `k=10`. 
# 
# 1. What is the index of the house in this query set that has the lowest predicted value? 
# 2. What is the predicted value of this house?

# In[ ]:

predictions = k_nn_predictions(10, features_train, output_train, features_test[0:10])
idx = np.argmin(predictions)
print idx
print predictions[idx]


# ## Choosing the best value of k using a validation set

# There remains a question of choosing the value of k to use in making predictions.
# Here, we use a validation set to choose this value. Write a loop that does the following:
# 
# * For `k` in [1, 2, ..., 15]:
#     * Makes predictions for each house in the VALIDATION set using the k-nearest neighbors from the TRAINING set.
#     * Computes the RSS for these predictions on the VALIDATION set
#     * Stores the RSS computed above in `rss_all`
# * Report which `k` produced the lowest RSS on VALIDATION set.

# (Depending on your computing environment, this computation may take 10-15 minutes.)

# In[ ]:

def choose_k(n):
    rss_all = np.zeros(n)
    for k in range(1,n+1):
        print k
        predictions = k_nn_predictions(k, features_train, output_train, features_valid)
        errors = predictions - output_valid
        rss_all[k-1] = np.dot(errors, errors)
    return rss_all

rss_all = choose_k(15)

# To visualize the performance as a function of `k`, plot the RSS on the VALIDATION set for each considered `k` value:

# In[ ]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

kvals = range(1, 16)
plt.plot(kvals, rss_all,'bo-')


# ***QUIZ QUESTION ***
# 
# What is the RSS on the TEST data using the value of k found above?  To be clear, sum over all houses in the TEST set.

# In[ ]:

k = 1 + np.argmin(rss_all)
predictions = k_nn_predictions(k, features_train, output_train, features_test)
errors = predictions - output_test
rss_test = np.dot(errors, errors)
print rss_test

