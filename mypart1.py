import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Function to read the features from file
def read_features(par_filename):
    vl = []
    with open(par_filename, "r") as file_lines:
        for line in file_lines:
            values = list(map(float, line.split()))
            del values[12]
            vl.append(values)
    return vl

# Function to read the labels from file
def read_labels(par_filename):
    vl = []
    with open(par_filename, "r") as file_lines:
        for line in file_lines:
            values = list(map(float, line.split()))
            vl.append(values[12])
    return vl

# Function to compute the classification using SVM
def compute_SVC(train_f, train_l):
    c = svm.SVC(kernel='linear')
    c.fit(train_f, train_l)
    return c

# Function to calculate the accuracy
def compute_accuracy(test_f, test_l, c):
    pred = c.predict(test_f)
    print(pred)
    pred_accu = accuracy_score(test_l, pred)
    return pred_accu

# Function to compute the confusion matrix
def compute_confusion_matrix(test_f, test_l, c):
    pred = c.predict(test_f)
    x = confusion_matrix(test_l, pred)
    return x

# Starting of the flow of program
read_data_features_train = read_features("plrx.txt")
read_data_labels_train = read_labels("plrx.txt")
model_svc = compute_SVC(read_data_features_train, read_data_labels_train)
accu_percent = compute_accuracy(read_data_features_train, read_data_labels_train, model_svc) * 100
print("Accuracy obtained over the whole training set is %0.6f %% ." % (accu_percent))