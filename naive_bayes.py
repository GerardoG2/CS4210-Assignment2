#-------------------------------------------------------------------------
# AUTHOR: Gerardo Gutierrez
# FILENAME: naive_bayes.py
# SPECIFICATION: Weather Classification
# FOR: CS 4210- Assignment #2
# TIME SPENT: 35 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X = []

outlook_dict     = {"Sunny":1, "Overcast":2, "Rain":3}
temp_dict        = {"Hot":1, "Mild":2, "Cool":3}
humidity_dict    = {"High":1, "Normal":2}
wind_dict        = {"Weak":1, "Strong":2}
class_dict       = {"Yes":1, "No":2}
inv_class_dict   = {1:"Yes", 2:"No"}

for row in dbTraining:
    X.append([
        outlook_dict[row[1]],
        temp_dict[row[2]],
        humidity_dict[row[3]],
        wind_dict[row[4]],
    ])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
y = []
for row in dbTraining:
    y.append(class_dict[row[5]])

#Fitting the naive bayes to the data using smoothing
clf = GaussianNB()

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
print(f"{'Day':<6}{'Outlook':<15}{'Temperature':<15}{'Humidity':<10}{'Wind':<15}{'PlayTennis':<15}{'Confidence'}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
clf.fit(X,y)

for data in dbTest:
    x_test = [[
        outlook_dict[data[1]],
        temp_dict[data[2]],
        humidity_dict[data[3]],
        wind_dict[data[4]],
    ]]

    probabilities = clf.predict_proba(x_test)[0]   
    prediction_index = probabilities.argmax() + 1         
    confidence = probabilities[prediction_index - 1]

    if confidence >= 0.75:
        print(f"{data[0]:<6}{data[1]:<15}{data[2]:<12}{data[3]:<15}{data[4]:<15}{inv_class_dict[prediction_index]:<15}{confidence:.2f}")



