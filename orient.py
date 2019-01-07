#!/usr/bin/env python3

import sys
import pandas as pd
import time
import numpy as np
import shutil
import math
from collections import defaultdict
import random
import pickle
from random import randrange
import operator

random.seed(42)

###############################################################################
# KNN
###############################################################################

#knnTrain method is used to generate model file. Since there is no actual
#training involved in the KNN classifier we are generating modelfile as a copy
#of the training file by using the python's shutil library.
def knnTrain(trainfile, modelfile):
  #  print ("TRAINING.......")
    shutil.copy(trainfile, modelfile)
  #  print ("TRAINING COMPLETED: MODEL FILE GENERATED")



#knnTest method reads the model file and test file using python's panda library and
#stores them as a dataframe called train and test respectively.
#trainorient: Orientations of the train data i.e. the second column in the train file.
#testid: Image IDs of the test data i.e. the first column in the test file.
#testorient: Image orientations of the test data i.e. the second column in the test file.
# After reading the files as a dataframe we perform indexing on the dataframe, delete the first
# and second columns of the files which are not initially required and  then convert the
# dataframes into numpy array i.e. testarray and trainarray containing only the pixel values
# All the data structures are then passed to knnCompute method which performs classification.
def knnTest(testfile, modelfile,model):
  #  print ("READING FILES.......")
    train = pd.read_table(modelfile, sep='\s+', header=None)
    test = pd.read_table(testfile, sep='\s+', header=None)

    trainorient = train.iloc[:][1].values
    testid = test.iloc[:][0].values
    testorient = test.iloc[:][1].values

    del train[0]
    del train[1]
    del test[0]
    del test[1]

    train.columns = range(train.shape[1])
    testarray = test.values
    trainarray = train.values
   # print ("FILES READ")

    knnCompute(testarray, trainarray, trainorient, testorient, testid,model)


# knnCompute method takes the input data structure from the knnTest method.
# Each test vector from the test data containing the pixels is then compared with each
# train vector and the corresponding euclidean distance is computed which is stored in
# distancelist[] for each test vector. We then sort the distance list and corresponding labels
# with respect to the distance. For the first K values we calculate the label with the highest
# frequency (VOTING). The label with the highest frequency is our classification for the respective test
# vector. This process is repeated for all the test vectors.
# We calculate the accuracy of our classification by comparing our solutionset[] with the correct labels in the
# test data and then give the accuracy in terms of the percentage.
# After numerous examinations we have come to the conclusion that the best value of K for our classifier is K=50.
def knnCompute(testarray, trainarray, trainorient, testorient, testid,model):
  #  print ("COMPUTING KNN........")
    solutionset = []
    for i in range(0, len(testarray)):
        testvector = testarray[i]
        t = []
        distancelist = []
        for trainvector in trainarray:
            distance = (np.sum(np.power((trainvector - testvector), 2)))
            distancelist.append(distance)
        t = tuple(zip(distancelist, trainorient))
        k = 50
        tups_deg = []
        B = sorted(t, key=lambda x: x[0])

        for j in range(0, k):
            tups_deg.append(B[j][1])
        mode = max(set(tups_deg), key=tups_deg.count)
        solutionset.append(mode)
    a = np.array(solutionset)
    b = np.array(testorient)
    count = np.sum(a == b)
    percent = (count / float(len(testorient))) * 100.0
    print ("Accuracy: {}".format(round(percent, 2)))
    writeFile(solutionset, testid,model)



# writeFile method is used to generate the output.txt file
# It takes the solutionset containing the lables and testid
# containing the imageIDs of the test data. It then combines them
# in a list usig zip method and then writes them onto a file.
def writeFile(solutionset, testid,model):
    s = []
    s = list(zip(testid, solutionset))
    model="output_"+model+".txt"
    with open(model, 'w') as fp:
        fp.write('\n'.join('%s %s' % tups for tups in s))


###############################################################################
# Adaboost
###############################################################################

#In this function we formulate the different hypothesis.
# The weak hypothesis is the comparison of the value a column with another coulmn
# If the first column isgreater it is assigned to the first class else it is assigned to the 2nd class

def hypothesis(data, p):
    r = [[]]
    for x in p:
        if data[x[0]] > data[x[1]]:
            r.append([x[0], x[1], -1])
        else:
            r.append([x[0], x[1], 1])
    r.pop(0)
    return (r)

#This is the function where the adaBoost is implemented 
#We find the different combination for the hypothesis(192C2) and select k(500) out of them
#Initialize the weight to 1/N
#Initialize a to 0
#Call the hypothesis function and store the value returned in a list, 
#the hypothesis classifies it into one of the 2 class
#REPEAT    
# Compare the value predicted by the hypothesis to the actual value
# If the prediction is worng -->error=error+w[j]    
#compute the error for all data in the training
#Now if the hypothesis and the actual label are correct reduce the weight of the actual by 
# w[m] = w[m]*(error/(1-error))   
#Then normalize the error
#update the value of a with  a[i]=math.log((1-error)/error)
#REPEAT the steps for all the hypothesis    

def adaBoost(data, label, l1, l2, k):
    N = len(data)
    w = [1 / N for i in range(N)]
    a = [0 for i in range(k)]
    p = [[]]
    sign = defaultdict(list)
    hyp = {}
    h = [[]]

    random.seed(103)
    feature = len(data[0])
    for x in range(0, feature):
        for y in range(x + 1, feature):
            p.append([x, y])
    p = random.sample(p, k)

    for j in range(0, N):
        r = hypothesis(data[j], p)
        hyp[j] = r
    for i in range(0, k):
        h.append(hyp[0][i][0:2])
    h.pop(0)
    for i in range(0, k):
        error = 0
        for j in range(0, N):
            if ((hyp[j][i][2] == 1 and label[j] == l1) or (hyp[j][i][2] == -1 and label[j] == l2)):
                error = error + w[j]
                sign[j].append((hyp[j][i][2] * -1))

        for m in range(0, N):
            if ((hyp[m][i][2] == 1 and label[m] == l2) or (hyp[m][i][2] == -1 and label[m] == l1)):
                w[m] = w[m] * (error / (1 - error))
                sign[m].append(hyp[j][i][2])

        sum1 = sum(w)
        w[:] = [x / sum1 for x in w]
        a[i] = math.log((1 - error) / error)
    return (a, h)

#This function does the actual testing of comparing the test image with the 
#weights and prediction done during the training    
#We do a one-one test herefor all the 6 combinations
#It puts the data to either of the 2 labels
#Finally mode is taken and then we choose the best one which classifies the data

def adaTest(data, weights, l1, l2, num):
    k = len(weights[num][0])
    N = len(data)
    a = weights[num][0]
    hyp = weights[num][1]
    h1 = [0 for i in range(N)]
    h = [0 for i in range(N)]
    sign = [[0 for x in range(N)] for y in range(k)]
    for i in range(0, k):
        for j in range(0, N):
            if (data[j][hyp[i][0]] > data[j][hyp[i][1]]):
                sign[i][j] = -1
            else:
                sign[i][j] = 1
    for i in range(0, k):
        for j in range(0, N):
            h[j] += a[i] * sign[i][j]
    for i in range(N):
        if h[i] < 0:
            h1[i] = l1
        else:
            h1[i] = l2
    return (h1)

#This function gets called if the modeis train.
#First we remove the 0th column, which is the name of the image
#we convert the data into 2 numpy array one containing the feature and the other only the label       
#we convert the train data into 6 different models
# 1. it contains only those data where the label is 0 or 90
# 2. it contains only those data where the label is 0 or 180
# 3. it contains only those data where the label is 0 or 270
# 4. it contains only those data where the label is 90 or 180
# 5. it contains only those data where the label is 90 or 270
# 6. it contains only those data where the label is 180 or 270    
# This function calls the adaBoost function for the different models
#Then it finally puts the output of the adaboost function to a dictionary and
#writes it to a pickle file    

def trainFunc(train, modelFile):
    k = 500
    train = pd.read_table(train, sep='\s+', header=None)
    labels = train[1]
    del train[0]

    sel1 = [0, 90]
    train1 = train.loc[train[1].isin(sel1)]
    label1 = train1[1]
    del train1[1]
    train1.columns = range(0, 192)
    train1 = train1.values
    label1 = label1.values

    sel2 = [0, 180]
    train2 = train.loc[train[1].isin(sel2)]
    label2 = train2[1]
    del train2[1]
    train2.columns = range(0, 192)
    train2 = train2.values
    label2 = label2.values

    sel3 = [0, 270]
    train3 = train.loc[train[1].isin(sel3)]
    label3 = train3[1]
    del train3[1]
    train3.columns = range(0, 192)
    train3 = train3.values
    label3 = label3.values

    sel4 = [90, 180]
    train4 = train.loc[train[1].isin(sel4)]
    label4 = train4[1]
    del train4[1]
    train4.columns = range(0, 192)
    train4 = train4.values
    label4 = label4.values

    sel5 = [90, 270]
    train5 = train.loc[train[1].isin(sel5)]
    label5 = train5[1]
    del train5[1]
    train5.columns = range(0, 192)
    train5 = train5.values
    label5 = label5.values

    sel6 = [180, 270]
    train6 = train.loc[train[1].isin(sel6)]
    label6 = train6[1]
    del train6[1]
    train6.columns = range(0, 192)
    train6 = train6.values
    label6 = label6.values

    del train[1]
    train.columns = range(0, 192)
    train = train.values

    finalDict = {}
    weight1, hyp1 = adaBoost(train1, label1, 0, 90, k)
    finalDict[1] = [weight1, hyp1]

    weight2, hyp2 = adaBoost(train2, label2, 0, 180, k)
    finalDict[2] = [weight2, hyp2]

    weight3, hyp3 = adaBoost(train3, label3, 0, 270, k)
    finalDict[3] = [weight3, hyp3]

    weight4, hyp4 = adaBoost(train4, label4, 90, 180, k)
    finalDict[4] = [weight4, hyp4]

    weight5, hyp5 = adaBoost(train5, label5, 90, 270, k)
    finalDict[5] = [weight5, hyp5]

    weight6, hyp6 = adaBoost(train6, label6, 180, 270, k)
    finalDict[6] = [weight6, hyp6]

    # modelFile=modelFile+".txt"

    with open(modelFile, 'wb') as handle:
        pickle.dump(finalDict, handle, )

#This function gets called if the modeis test.
#First we remove the 0th column, which is the name of the image
#we convert the data into 2 numpy array one containing the feature and the other only the label 
#We call the adaTest function for each of the 6 types of combinations made during the training
#the output of the adaTest function is a list which contains the predicted value for that combination
#Finally we take the mode of these output and that will be the final prediction
#Accuracy is taken by comparing the prdiced value with the orginal value   

def testFunc(test, modelFile,model):
    test = pd.read_table(test, sep='\s+', header=None)
    with open(modelFile, 'rb') as handle:
        finalDict = pickle.load(handle)

    testLabel = test[1]
    testid = test[0]
    del test[0]
    del test[1]
    test.columns = range(0, 192)
    test = test.values
    testLabel = testLabel.values

    h1 = adaTest(test, finalDict, 0, 90, 1)
    h2 = adaTest(test, finalDict, 0, 180, 2)
    h3 = adaTest(test, finalDict, 0, 270, 3)
    h4 = adaTest(test, finalDict, 90, 180, 4)
    h5 = adaTest(test, finalDict, 90, 270, 5)
    h6 = adaTest(test, finalDict, 180, 270, 6)

    pred = []
    pred = [max(set(i), key=i.count) for i in zip(h1, h2, h3, h4, h5, h6)]

    c = 0
    for i in range(len(testLabel)):
        if testLabel[i] == pred[i]:
            c += 1

        accuracy = c * 100 / len(testLabel)
    print("accuracy:", accuracy)

    writeFile(pred, testid,model)


###############################################################################
# Random Forest
###############################################################################

# Reading train data
def build_train_data(train):
    train_df = pd.read_table(train, sep='\s+', header=None)

    files_train = train_df[0]

    labels_train = train_df[1]

    del train_df[0]
    del train_df[1]

    # Adding labels to last
    train_df = pd.concat([train_df, labels_train], axis=1)

    train_df.columns = range(0, 193)

    # Creating a numpy array from the pandas dataset
    train = train_df.values

    labels_uniq = list(set(label for label in labels_train))

    return train_df, train, labels_uniq


# Reading test data
def build_test_data(test):
    test_df = pd.read_table(test, sep='\s+', header=None)

    files_test = test_df[0]

    labels_test = test_df[1]

    del test_df[0]
    del test_df[1]

    test = test_df.values

    return test_df, test, labels_test, files_test


def unique_values(rows, col):
    return set([row[col] for row in rows])


# Finding the counts of labels for a given set of rows
def label_count(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# This class is used to store the column and the threshold value for maximum information gain
# and also to return 'True' if the value of the selected column in a row is greater than the
# thresold and 'False' if the value is less
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        try:
            val = example[self.column]
        except:
            return False
        return val >= self.value


# This functions divides the rows based the value in the column, if the value if greater than
# the threshold then the row is labellled 'true_row' else it is labelled 'false_row'
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# This function is used to calculate gini impurity for a set of rows using the formula:
# gini = 1-sum over all labels(count of label/count of rows)^2
def gini(rows):
    counts = label_count(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


# This function is used to calculate the information gain for the current thresold as follows:
# 1. gini impurity obtained for the current set of true and false rows is calculated
# 2. the impurity is multiplied with the corresponding weights which is the proportion of rows
#   in each set
# 3. The total impurity is subtracted is from the previous uncertainity
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


# This function returns the value of best gain and corresponding threshold(row, column value)
# based on the highest information gain
def find_best_split(rows, features):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    # n_features = len(rows[0]) - 1
    # print('features:',features)

    for col in features:  # range(n_features):

        col_values = [row[col] for row in rows]
        values = [np.percentile(col_values, 25), np.median(col_values), np.percentile(col_values, 75)]

        val = np.median(col_values)
        # for val in col_values:  # for each value

        question = Question(col, val)

        # try splitting the dataset
        true_rows, false_rows = partition(rows, question)

        if len(true_rows) == 0 or len(false_rows) == 0:
            continue

        # Calculate the information gain from this split
        gain = info_gain(true_rows, false_rows, current_uncertainty)

        if gain > best_gain:
            best_gain, best_question = gain, question

    return best_gain, best_question


# This class is used to count the number of labels at the leaf node for a set of rows
class Leaf:
    def __init__(self, rows):
        self.predictions = label_count(rows)


# This class is used to store the information for a given node, i.e. the thresold at
# which the division has to be made, the true rows and false rows
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# This function is used to build a tree based on parameters given, the parameters
# chosen are-
# 1. max depth - the depth to which the nodes are split
# 2. depth - the current depth upto which the nodes are split
# 3. features - the number of features from the full set of 192 features to consider
#   to build the tree
# 4. row - the rows to consider to build the tree
# This function finds the best for the given set of rows and parameters and if the
# gain is 0 then the functions return the set of rows as leaf nodes, if not then the
# function finds the true and false rows based on the best split value. Now if the max
# depth is reached then the rows are return as leaf nodes.
# The function calls itself recursively to build the tree if none of the two conditions
# are satisfied. Finally the value of decision nodes which are the nodes at which the
# decision tree classifies the input row into true branch or false branch
def build_tree(rows, max_depth, depth, features):
    gain, question = find_best_split(rows, features)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    if depth > max_depth:
        return Leaf(rows)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, max_depth, depth + 1, features)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, max_depth, depth + 1, features)

    return Decision_Node(question, true_branch, false_branch)


# this function is used to transverse a tree based on the value for a column of that
# at each node until a leaf node is reached then the predictions for that row are
# returned once the leaf node is reached
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# This functions returns the predicted class after obtaining the predictions using the
# 'classify' function. This is done by return the class with max count for a given
# prediction
def predict(tree, row):
    leaf_labels = classify(row, tree)
    return max(leaf_labels.items(), key=operator.itemgetter(1))[0]


# This function is used to make subsets of the dataset to build trees on
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Function to predict class for a row by taking the vote from each of the built trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest
# Function to build a forest of trees based on the parameters passed. The parameter
# passed are as follows:
# 1. max_depth - the depth to which the nodes are split
# 2. sample_size_ratio - the ratio of the dataset which is used to build trees of
#   the forest
# 3. n_tress - number of trees which are build for the forest
# 4. n_features - the number of features from the full set of 192 features to consider
#   to build the tree
def random_forest(train, max_depth, sample_size_ratio, n_trees, n_features):
    train_df, train, labels_uniq = build_train_data(train)

    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size_ratio)
        features = []
        while len(features) < n_features:
            col_index = randrange(len(train[0] - 1))
            if col_index not in features:
                features.append(col_index)
        # print(features)
        tree = build_tree(sample, 3, 1, features)
        trees.append(tree)
    return trees


def pred_rotation(test, trees,model):
    test_df, test, labels_test, files_test = build_test_data(test)

    predictions = [bagging_predict(trees, row) for row in test]

    writeFile(predictions, files_test,model)

    acc = 0
    for i in range(len(predictions)):
        if predictions[i] == labels_test[i]:
            acc += 1
    acc = acc / len(predictions)

    print('accuracy:', acc * 100)
    return predictions


###############################################################################
# Execution
###############################################################################
option = sys.argv[1]
optionfile = sys.argv[2]
modelfile = sys.argv[3]
model = sys.argv[4]

# print(model)
start_time=time.time()
if model == 'nearest':
    if option == "train":
        knnTrain(optionfile, modelfile)
    elif option == "test":
        knnTest(optionfile, modelfile,model)
   # print ("Time Taken: {} minutes".format((time.time()-start_time)/60.0))

elif model == 'adaboost' or model =='best':
    if option == "train":
        trainFunc(optionfile, modelfile)
    elif option == "test":
        testFunc(optionfile, modelfile,model)

    end = time.time()
   # print("time",end-start_time)

elif model == 'forest':
    if option == 'train':
        trees = random_forest(optionfile, 4, 0.7, 50, 20)  # Using the best and optimum features
        with open(modelfile, 'wb') as handle:
            pickle.dump(trees, handle, )
    elif option == 'test':
        with open(modelfile, 'rb') as handle:
            trees = pickle.load(handle)
            pred_label = pred_rotation(optionfile, trees,model)
    end = time.time()
    #print("time",end-start_time)

#elif model == 'best':
#    if option == "train":
#        knnTrain(optionfile, modelfile)
#    elif option == "test":
#        knnTest(optionfile, modelfile)
#    print ("Time Taken: {} minutes".format((time.time()-start_time)/60.0))