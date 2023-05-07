import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Function to read the data from file
def read_data(par_filename):
    vl = []
    with open(par_filename, "r") as file_lines:
        for line in file_lines:
            vl.append(list(map(float, line.split())))
    return vl

# Function to read the labels from file
def read_labels(vl):
    ll = []
    for r in vl:
        ll.append(int(r[12]))
    return ll

# Function to read the features from file
def read_features(vl):
    lp = vl.copy()
    for r in lp:
        r.pop(12)
    return lp

# Function to compute the classification using SVM
def compute_SVC(train_f, train_l):
    c = svm.SVC(kernel='linear')
    c.fit(train_f, train_l)
    return c

# Function to calculate the accuracy
def compute_accuracy(test_f, test_l, c):
    pred = c.predict(test_f)
    pred_accu = accuracy_score(test_l, pred)
    return pred_accu

# Function to compute the confusion matrix
def compute_confusion_matrix(test_f, test_l, c):
    pred = c.predict(test_f)
    x = confusion_matrix(test_l, pred)
    return x

# Function to compute the error
def compute_error(t_f, t_l, c):
    err = c.score(t_f, t_l)
    return err

# Function to split the data based on percentage
def split_data(f, percent):
    tot = len(f)
    req_xt = int((float(percent) / 100) * tot)
    req_yt = tot - req_xt
    xt_get = f[:req_xt]
    yt_get = f[req_xt:]
    xyt = [xt_get, yt_get]
    return xyt

# Function to plot the training and testing errors
def compute_plot(filename):
    test_plt = []
    train_plt = []
    percent_plt = []
    with open(filename, "r") as lines_in_file:
        for line in lines_in_file:
            values = line.split()
            test_plt.append(float(values[0]))
            train_plt.append(float(values[1]))
            percent_plt.append(float(values[2]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(percent_plt, test_plt, 'bo', label='Training Error')
    plt.plot(percent_plt, train_plt, 'ro', label='Testing Error')
    plt.plot(percent_plt, test_plt, 'b')
    plt.plot(percent_plt, train_plt, 'r')
    ax.set_xlabel('Percentage of Training data')
    ax.set_ylabel('Percentage of Error')
    plt.legend(loc='upper left', numpoints=1)
    plt.title("% Error Vs % training Data")
    plt.show()
    return

# Starting of the flow of program
read_data = read_data("plrx.txt")
read_data_labels = read_labels(read_data)
read_data_features = read_features(read_data)
input_percent = [40, 50, 60, 70, 80, 90]
with open('Generated_accuracy_table.dat', 'w') as file_created1, open('Generated_error_table.dat', 'w') as file_created2:
    for pri in input_percent:
        x1 = split_data(read_data_features, pri)
        x2 = split_data(read_data_labels, pri)
        train_features = x1[0]
        train_labels = x2[0]
        test_features = x1[1]
        test_labels = x2[1]
        model_svc = compute_SVC(train_features, train_labels)
        accu_percent_train = compute_accuracy(train_features, train_labels, model_svc) * 100
        accu_percent_test = compute_accuracy(test_features, test_labels, model_svc) * 100
        train_err = compute_error(train_features, train_labels, model_svc)
        test_err = compute_error(test_features, test_labels, model_svc)
        file_created1.write("%f %f %f\n" % (accu_percent_train, accu_percent_test, pri))
        file_created2.write("%f %f %f\n" % (train_err, test_err, pri))
        conf_mat = compute_confusion_matrix(train_features, train_labels, model_svc)
        print(conf_mat)
        conf_mat1 = compute_confusion_matrix(test_features, test_labels, model_svc)
        print(conf_mat1)

compute_plot("Generated_accuracy_table.dat")
compute_plot("Generated_error_table.dat")
