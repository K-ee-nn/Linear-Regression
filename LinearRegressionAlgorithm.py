import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
#-----------------------------------------------------------------------------------------------------
# Reading in and Looking at Data
#-----------------------------------------------------------------------------------------------------
                   # Math class
data = pd.read_csv("student-mathclass.csv", sep=';') # Read the csv in

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']] # Create an array from the csv

predict = 'G3' # G3 is what we want to predict
#-----------------------------------------------------------------------------------------------------
# Creating and implementing the Model
#-----------------------------------------------------------------------------------------------------
X = np.array(data.drop([predict], 1)) # array 1
y = np.array(data[predict])         # array 2
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# test X and y based on x_train, y_train, x_test, y_test

#-------------------------------------------------------------------------------------------------------
# Always save the best model results
#-------------------------------------------------------------------------------------------------------
"""best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1 )
    

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best: 
        best = acc # Replace previous result by new result if acc is higher
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)"""



read_pickle_in = open("studentmodel.pickle", 'rb')
linear = pickle.load(read_pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


#-----------------------------------------------------------
# Plotting using matplotlib
#-----------------------------------------------------------

p = 'G1'
style.use('ggplot')
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()