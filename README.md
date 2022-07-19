# Rmove-method
/json : Project dependency using depends</br>
/moved_method : Recommended method of moving</br>
/src : The code files which is involved in the experiment</br>
/Training_CSV_code2seq : The training data of feature fusion with code2seq and network embedding</br>
/Training_CSV_code2vec : The training data of feature fusion with code2vec and network embedding</br>


#Introduction
This document is included in the  'Recommending Move Method Refactoring Opportunities using Structural and Semantic Representations of Code' distribution，which we will refer to as  RMove，This is to distinguish the recommended implementation of this move method refactoring from other implementations.
In this document, the environment required to make and use the Rmove tool is described. Some hints about the installation environment are here, but users need to find complete instructions from other sources. They give a more detailed description of their tools and instructions for using them.
My main environment is located on a computer with windows (windows 10 at the time of my writing) operating system. The fundamentals should be similar for other platforms, although the way in which the environment is configured will be different.
What do I mean by environment? For example, to run python code you will need to install a python interpreter, and if you want to use code2vec you will need tensorflow.

# tools

## code embedding network

code2vec
code2seq

## graph embedding network

DeepWalk
LINE
Node2vec
GraRep
SDNE
ProNE
walklets
#Requirement

###code2vec

Python3 (>=3.6). To check the version:

TensorFlow - version 2.0.0 (install). To check TensorFlow version:

If you are using a GPU, you will need CUDA 10.0 (download) as this is the version that is currently supported by TensorFlow. To check CUDA version:

nvcc --version

For GPU: cuDNN (>=7.5) (download) To check cuDNN version:

code2seq

Python3 (>=3.6). To check the version:

TensorFlow - version 2.0.0 (install). To check TensorFlow version:

If you are using a GPU, you will need CUDA 10.0 (download) as this is the version that is currently supported by TensorFlow. To check CUDA version:

nvcc --version

For GPU: cuDNN (>=7.5) (download) To check cuDNN version:

###DeepWalk，LINE ，Node2vec，GraRep，SDNE
They all come from the open source project OpenNE

numpy==1.14
networkx==2.0
scipy==0.19.1
tensorflow>=1.12.1
gensim==3.0.1
scikit-learn==0.19.0

###ProNE

numpy
sklearn
networkx
gensim

###walklets

tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
gensim            3.6.0
networkx          2.4

#quickstart

>step1:Use code2vec and code2seq to embedding the semantic information of the code and get the vectors corresponding to the methods and classes

eg:
feature1   feature2   feature3 ......    featuren
0.98         0.16          0.98     .......     0.87
>step 2:We use graph embedding networks such as DeepWalk to embedding the structural information of the code dependencies and get the structural vector of the method dependency graph

eg:
feature1   feature2      feature3     ......       featuren
0.98             0.16             0.98     .......     0.87

>step3 : We train the training set using classifiers commonly used in machine learning and deep learning, and optimize the hyperparameters using grid search

>step4 : Model evaluation on the real-world dataset

#datasets
train data
PMD
Cayenne
Pinpoint
Jenkins
Drools

real world data
Weka
Ant
FreeCol
JMeter
FreeMind
JTOpen
DrJava
Maven

>We generate training data from a small set of detected move method detection results. the paper presents the procedure of training data generation.The input of this algorithm is a set of detected move method results: MoveMethodSet. This algorithm inspects each item in MoveMethodSet iteratively.
