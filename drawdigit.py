
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib #This module helps with offlining classifier
from sklearn import cross_validation #This module helps with generalization of our classifier
from sklearn.grid_search import GridSearchCV #This module uses grid search to cross validate classifier

#Importing data
#Name:    loadData(location)
#Purpose: Loads in data from directory specified by location parameter
#Return:  If training data, returns labels and data. If testing data,
#         returns just testing data (all in numpy array format)
def loadData(location):
 print 'Accessing: ' + location
 df = pd.read_csv(location)
 #Pandas allows conversion from dataframe to numpy array using df.values
 data = df.values
 #Extracting labels from the first column of the training data.
 if 'train' in location:
  #data = data[0:5000,:]
  labels = data[:,0]
  size = data.shape[1]
  #Removing labels because we don't want to use our answers as one of the features!
  data = data[:,1:size]
  return labels, data
 else:
  return data

def train_svm(train, labels):
 #Using exhaustive grid search to find best hyperparameter combination
 tuning_params = [
                  {'kernel': ['rbf'], 'gamma': [1e-2, 1e-1], 'C': [0.1, 1.0]},
                  {'kernel': ['poly'], 'degree': [3, 4], 'C': [0.1, 1.0]}
                 ]
 print 'Tuning hyperparameters for accuracy (%) ... \n'
 #Using support vector classifier
 model = GridSearchCV(SVC(C=0.1), tuning_params, cv=3)
 model.fit(train, labels)
 print 'Best parameters found: \n'
 print (model.best_params_)
 print 'Highest accuracy of grid search: \n'
 print (model.best_score_)
 print 'Summary of grid search: \n'
 print (model.grid_scores_)
 #Final trained model
 return model

def main():
 trainLoc = 'data/train.csv'
 testLoc = 'data/test.csv'

 labels, trainingData = loadData(trainLoc)
 testingData = loadData(testLoc)

 classifier = raw_input('Would you like to train (1) or test (2)?: ')
 if int(classifier) == 1:
  print 'Training using SVM'
  model = train_svm(trainingData, labels)
  joblib.dump(model, 'svm_digit.pkl')
  print 'Done!'

#Standard boilerplate check when program is called from command line
if __name__ == '__main__':
 main()