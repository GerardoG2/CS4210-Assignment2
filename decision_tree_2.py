#-------------------------------------------------------------------------
# AUTHOR: Gerardo Gutierrez
# FILENAME: decision_tree_2.py
# SPECIFICATION: Compare 3 Decision Tree Models
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    dbTraining = pd.read_csv(ds).values.tolist()

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    age_dict        = {"Young":0, "Prepresbyopic":1, "Presbyopic":2}
    spectacle_dict       = {"Myope":0, "Hypermetrope":1}
    astigmatism_dict      = {"No":0, "Yes":1}
    tear_dict      = {"Reduced":0, "Normal":1}
    

    for row in dbTraining:
        age = row[0]
        spectacle = row[1]
        astigmatism = row[2]
        tear = row[3]

        X.append([
            age_dict[age],
            spectacle_dict[spectacle],
            astigmatism_dict[astigmatism],
            tear_dict[tear],
        ])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    recommended_dict    = {"No":0, "Yes":1} 
    for row in dbTraining:

        recommended = row[4]

        Y.append([
        recommended_dict[recommended]
        ])



    #Loop your training and test tasks 10 times here
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5

       clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=i)
       clf.fit(X,Y)

         # initialize total accuracy of model
       if i == 0: 
           total_accuracy = 0 
       
       # initialize variables for number of accurate predictions and total tests
       n_correct = 0
       n_tests = 0

       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           age = data[0]
           spectacle = data[1]
           astigmatism = data[2]
           tear = data[3]

           x_test = [[
               age_dict[age],
               spectacle_dict[spectacle],
               astigmatism_dict[astigmatism],
               tear_dict[tear]
           ]]

           class_predicted = clf.predict(x_test)[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           true_label = recommended_dict[data[4]]
           if class_predicted == true_label:
               n_correct += 1
           n_tests += 1

       if n_tests > 0: 
        total_accuracy += n_correct / n_tests            

    #Find the average of this model during the 10 runs (training and test set)
    average_accuracy = total_accuracy / 10

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f'\nTraining Set: {ds}\nAccuracy:{average_accuracy}')




