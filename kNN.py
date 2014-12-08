## kNN.py
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

## This function takes four inputs: vector to classify called inX
## the full matrix of training examples dataSet
## a vector of labels called labels
## and k, the number of nearest neighbors to use in the voting
## length(labels) must be equal to nrow(dataSet)

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
### Method outline:
## Euclidean distance is computed, the first k orlowest k are used to vote on the ## class of inX, the callCount is taken (as a dict) and decomposed into a list of
## tuples. The tuples are then sorted by the second item using the itemgetter()
## method from the operator module. Reverse sort to return most freq first. 

### This function takes the filename string and outputs two things, a matrix of
##  training examples and a vector of class lables
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

### Mehod Outline:
## Reads the file and counts the number of lines
## Create a NumPy matrix to populate and return (essentially a 2-D array)
## Finally loops over the lines in the file and strip off the return line char
## with line.strip(). Next split it into a list of elements delimited by '\t'
## Take the first three elements into the matrix, row-wise
## then use negative indexing to get the last item into classLabelVector 







