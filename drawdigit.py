
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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
  labels = data[:,0]
  size = data.shape[1]
  #Removing labels because we don't want to use our answers as one of the features!
  data = data[:,1:size]
  return labels, data
 else:
  return data

def knn(train, labels, test):
 knn = KNeighborsClassifier()
 knn.fit(train, labels)
 predictions = knn.predict(test)
 return predictions

def svm(train, labels, test):
 svm_poly = SVC(kernel='poly', degree=4)
 #svm_poly = svm.SVC(kernel='poly', degree=4)
 svm_poly.fit(train, labels)
 predictions = svm_poly.predict(test)
 return predictions

def main():
 trainLoc = 'data/train.csv'
 testLoc = 'data/test.csv'

 labels, trainingData = loadData(trainLoc)
 testingData = loadData(testLoc)

 classifier = raw_input('Which classifier would you like to use? \n 1: K-NN \n 2: SVM \n Your selection: ')
 if int(classifier) == 1:
  print 'Training and classifying using K Nearest Neighbors'
  predictions = knn(trainingData, labels, testingData)
  np.savetxt('predictions_knn.csv', predictions, delimiter=',')
  print 'Done!'

 elif int(classifier) == 2:
  print 'Training and classifying using SVM with polynomial Kernel'
  predictions = svm(trainingData, labels, testingData)
  np.savetxt('predictions_svm.csv', predictions, delimiter=',')
  print 'Done!'

if __name__ == '__main__':
 main()