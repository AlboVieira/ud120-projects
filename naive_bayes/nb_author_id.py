#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

labels = ["Sara", "Chris"]

#########################################################
### your code goes here ###
training_time = time()
clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
print "Time to training:", round(time()-training_time, 3), "s"

testing_time = time()
pred = clf.predict(features_test)
print "Time to test:", round(time()-testing_time, 3), "s"
print pred

#########################################################


