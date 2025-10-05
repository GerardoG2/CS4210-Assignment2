#-------------------------------------------------------------------------
# AUTHOR: Gerardo Gutierrez
# FILENAME: knn.py
# SPECIFICATION: LOO CV Error Rate for 1nn
# FOR: CS 4210- Assignment #2
# TIME SPENT: 40 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

correct_predictions = 0
total_instances = len(db)
#Loop your data to allow each instance to be your test set
for i in db:

    X = []
    Y = []

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    for j in db:
        if j == i:
            continue
        X.append([float(value) for value in j[:-1]])

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    for j in db: 
        if j == i:
            continue
        if j[-1].lower() == 'spam':
            Y.append(1)
        else:
            Y.append(0)

    #Store the test sample of this iteration in the vector testSample
    testSample = [float(value) for value in i[:-1]]
    if i[-1].lower() == "spam":
        true_label = 1
    else: true_label = 0

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    clf = KNeighborsClassifier(n_neighbors=1, metric = 'minkowski', p=2)
    clf.fit(X,Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted == true_label:
        correct_predictions += 1

#Print the error rate
error_rate = 1 - correct_predictions / total_instances

print(f'LOO-CV error rate for 1nn classifier: {error_rate:.4f}')






