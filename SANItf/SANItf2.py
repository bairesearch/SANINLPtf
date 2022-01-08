"""SANItf2.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and Tensorflow 2.1+ 

conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
conda install scikit-learn (SANItf2_algorithmLIANN_PCAsimulation only)
	
# Usage:
python3 SANItf2.py

# Description:
SANItf - train an experimental artificial neural network (SANI)

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

from ANNtf2_operations import *
import ANNtf2_globalDefs
from numpy import random
import ANNtf2_loadDataset
from SANItf2_algorithmSANIglobalDefs import algorithmSANI

#select algorithm:
algorithm = "SANI"	#sequentially activated neuronal input artificial neural network	#incomplete+non-convergent

suppressGradientDoNotExistForVariablesWarnings = True

costCrossEntropyWithLogits = False
if(algorithm == "SANI"):
	#set algorithmSANI in SANItf2_algorithmSANIoperations
	if(algorithmSANI == "sharedModulesHebbian"):
		import SANItf2_algorithmSANIsharedModulesHebbian as SANItf2_algorithm
		#no cost function used
	elif(algorithmSANI == "sharedModulesBinary"):
		import SANItf2_algorithmSANIsharedModulesBinary as SANItf2_algorithm
	elif(algorithmSANI == "sharedModules"):
		import SANItf2_algorithmSANIsharedModules as SANItf2_algorithm
		costCrossEntropyWithLogits = True
	elif(algorithmSANI == "repeatedModules"):
		import SANItf2_algorithmSANIrepeatedModules as SANItf2_algorithm
	
						
#learningRate, trainingSteps, batchSize, displayStep, numEpochs = -1

#performance enhancements for development environment only: 
debugUseSmallPOStagSequenceDataset = True	#def:False	#switch increases performance during development	#eg data-POStagSentence-smallBackup
useSmallSentenceLengths = True	#def:False	#switch increases performance during development	#eg data-simple-POStagSentence-smallBackup
trainMultipleFiles = False	#def:True	#switch increases performance during development	#eg data-POStagSentence
trainMultipleNetworks = False	#trial improve classification accuracy by averaging over multiple independently trained networks (test)
numberOfNetworks = 1

if(trainMultipleFiles):
	fileIndexFirst = 0
	if(useSmallSentenceLengths):
		fileIndexLast = 11
	else:
		fileIndexLast = 1202
				
#loadDatasetType3 parameters:
#if generatePOSunambiguousInput=True, generate POS unambiguous permutations for every POS ambiguous data example/experience
#if onlyAddPOSunambiguousInputToTrain=True, do not train network with ambiguous POS possibilities
#if generatePOSunambiguousInput=False and onlyAddPOSunambiguousInputToTrain=False, requires simultaneous propagation of different (ambiguous) POS possibilities

if(algorithm == "SANI"):
	if(SANItf2_algorithm.algorithmSANI == "sharedModulesHebbian"):
		if(SANItf2_algorithm.SANIsharedModules):	#optional
			dataset = "POStagSentence"
			numberOfFeaturesPerWord = -1
			paddingTagIndex = -1
			generatePOSunambiguousInput = False
			onlyAddPOSunambiguousInputToTrain = False	#True
		else:
			print("!SANItf2_algorithm.SANIsharedModules")
			dataset = "POStagSequence"
	elif(SANItf2_algorithm.algorithmSANI == "sharedModulesBinary"):
		if(SANItf2_algorithm.SANIsharedModules):	#only implementation
			print("sharedModulesBinary")
			dataset = "POStagSentence"
			numberOfFeaturesPerWord = -1
			paddingTagIndex = -1	
			generatePOSunambiguousInput = False
			onlyAddPOSunambiguousInputToTrain = False	#True
		else:
			print("SANItf2 error: (SANItf2_algorithm.algorithmSANI == sharedModulesBinary) && !(SANItf2_algorithm.SANIsharedModules)")
			exit()
	elif(SANItf2_algorithm.algorithmSANI == "sharedModules"):
		if(SANItf2_algorithm.SANIsharedModules):	#only implementation
			dataset = "POStagSentence"
			numberOfFeaturesPerWord = -1
			paddingTagIndex = -1
			if(SANItf2_algorithm.allowMultipleContributingSubinputsPerSequentialInput):
				generatePOSunambiguousInput = False
				onlyAddPOSunambiguousInputToTrain = False
			else:
				generatePOSunambiguousInput = False
				onlyAddPOSunambiguousInputToTrain = True
		else:
			print("SANItf2 error: (SANItf2_algorithm.algorithmSANI == sharedModules) && !(SANItf2_algorithm.SANIsharedModules)")
			exit()
	elif(SANItf2_algorithm.algorithmSANI == "repeatedModules"):
		if(not SANItf2_algorithm.SANIsharedModules):	#only implementation
			dataset = "POStagSequence"
		else:
			print("SANItf2 error: (SANItf2_algorithm.algorithmSANI == repeatedModules) && (SANItf2_algorithm.SANIsharedModules)")	
			exit()		
		
			
if(debugUseSmallPOStagSequenceDataset):
	dataset1FileNameXstart = "Xdataset1PartSmall"
	dataset1FileNameYstart = "Ydataset1PartSmall"
	dataset3FileNameXstart = "Xdataset3PartSmall"
else:
	dataset1FileNameXstart = "Xdataset1Part"
	dataset1FileNameYstart = "Ydataset1Part"
	dataset3FileNameXstart = "Xdataset3Part"
datasetFileNameXend = ".dat"
datasetFileNameYend = ".dat"
datasetFileNameStart = "datasetPart"
datasetFileNameEnd = ".dat"



def defineTrainingParameters(dataset, numberOfFeaturesPerWord=None, paddingTagIndex=None):
	SANItf2_algorithm.defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWord, paddingTagIndex)
	return SANItf2_algorithm.defineTrainingParametersSANIwrapper(dataset, trainMultipleFiles)

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths=None, numberOfFeaturesPerWord=None):
	return SANItf2_algorithm.defineNetworkParametersSANIwrapper(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, useSmallSentenceLengths, numberOfFeaturesPerWord)

def defineNeuralNetworkParameters():
	return SANItf2_algorithm.defineNeuralNetworkParameters()

#define default forward prop function for test (identical to below);
def neuralNetworkPropagationTest(test_x, networkIndex=1):
	return SANItf2_algorithm.neuralNetworkPropagation(test_x, networkIndex)

#define default forward prop function for backprop weights optimisation;
def neuralNetworkPropagation(x, networkIndex=1, l=None):
	return SANItf2_algorithm.neuralNetworkPropagation(x, networkIndex)


def trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display, l=None):
	
	if(algorithm == "SANI"):
		#learning algorithm not yet implemented:
		#if(batchSize > 1):
		pred = neuralNetworkPropagation(batchX)	
		print("pred = ", pred)		

	pred = None
			
	return pred
			
	
def loadDataset(fileIndex):

	global numberOfFeaturesPerWord
	global paddingTagIndex
	global numberOfFeaturesPerWord
	global numberOfFeaturesPerWord
	global numberOfFeaturesPerWord
	
	datasetNumFeatures = 0
	datasetNumClasses = 0
	
	fileIndexStr = str(fileIndex).zfill(4)
	if(dataset == "POStagSequence"):
		datasetType1FileNameX = dataset1FileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = dataset1FileNameYstart + fileIndexStr + datasetFileNameYend
	elif(dataset == "POStagSentence"):
		datasetType3FileNameX = dataset3FileNameXstart + fileIndexStr + datasetFileNameXend		
	elif(dataset == "SmallDataset"):
		if(trainMultipleFiles):
			datasetType2FileName = dataset2FileNameStart + fileIndexStr + datasetFileNameEnd
		else:
			datasetType2FileName = dataset2FileName

	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	
	return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp


#trainMinimal is minimal template code extracted from train based on trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False;
def trainMinimal():
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		fileIndex = 0
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

		shuffleSize = datasetNumExamples	#heuristic: 10*batchSize
		trainDataIndex = 0

		trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
		trainDataList = []
		trainDataList.append(trainData)
		trainDataListIterators = []
		for trainData in trainDataList:
			trainDataListIterators.append(iter(trainData))

		for batchIndex in range(int(trainingSteps)):
			(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
			batchYactual = batchY
					
			display = False
			if(batchIndex % displayStep == 0):
				display = True	
			trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display)

		pred = neuralNetworkPropagationTest(test_x, networkIndex)
		print("Test Accuracy: %f" % (calculateAccuracy(pred, test_y)))

			
#this function can be used to extract a minimal template (does not support algorithm==LREANN);
def train(trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False):
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()

	#configure optional parameters;
	if(trainMultipleNetworks):
		maxNetwork = numberOfNetworks
	else:
		maxNetwork = 1
	if(trainMultipleFiles):
		minFileIndex = fileIndexFirst
		maxFileIndex = fileIndexLast
	else:
		minFileIndex = 0
		maxFileIndex = 0
	if(greedy):
		maxLayer = numberOfLayers
	else:
		maxLayer = 1
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		#fileIndex = 0
		#trainMultipleFiles code;
		if(trainMultipleFiles):
			fileIndexArray = np.arange(fileIndexFirst, fileIndexLast+1, 1)
			#print("fileIndexArray = " + str(fileIndexArray))
			np.random.shuffle(fileIndexArray)
			fileIndexShuffledArray = fileIndexArray
			#print("fileIndexShuffledArray = " + str(fileIndexShuffledArray))
		for fileIndex in range(minFileIndex, maxFileIndex+1):

			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

			shuffleSize = datasetNumExamples	#heuristic: 10*batchSize
			trainDataIndex = 0

			#greedy code;
			for l in range(1, maxLayer+1):
				print("l = ", l)
				trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
				trainDataList = []
				trainDataList.append(trainData)
				trainDataListIterators = []
				for trainData in trainDataList:
					trainDataListIterators.append(iter(trainData))

				for batchIndex in range(int(trainingSteps)):
					(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
					batchYactual = batchY
					
					#trainMultipleNetworks code;
					predNetworkAverage = tf.Variable(tf.zeros(datasetNumClasses))
					for networkIndex in range(1, maxNetwork+1):

						display = False
						#if(l == maxLayer):	#only print accuracy after training final layer
						if(batchIndex % displayStep == 0):
							display = True	
						pred = trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display, l)
						if(pred is not None):
							predNetworkAverage = predNetworkAverage + pred
						
					#trainMultipleNetworks code;
					if(trainMultipleNetworks):
						if(display):
							predNetworkAverage = predNetworkAverage / numberOfNetworks
							loss = calculateLossCrossEntropy(predNetworkAverage, batchYactual, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(predNetworkAverage, batchYactual)
							print("batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))	

				#trainMultipleNetworks code;
				if(trainMultipleNetworks):
					predNetworkAverageAll = tf.Variable(tf.zeros([test_y.shape[0], datasetNumClasses]))
					for networkIndex in range(1, numberOfNetworks+1):
						pred = neuralNetworkPropagationTest(test_x, networkIndex)	#test_x batch may be too large to propagate simultaneously and require subdivision
						print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
						predNetworkAverageAll = predNetworkAverageAll + pred
					predNetworkAverageAll = predNetworkAverageAll / numberOfNetworks
					#print("predNetworkAverageAll", predNetworkAverageAll)
					acc = calculateAccuracy(predNetworkAverageAll, test_y)
					print("Test Accuracy: %f" % (acc))
				else:
					pred = neuralNetworkPropagationTest(test_x, networkIndex)
					if(greedy):
						print("Test Accuracy: l: %i, %f" % (l, calculateAccuracy(pred, test_y)))
					else:
						print("Test Accuracy: %f" % (calculateAccuracy(pred, test_y)))

						

				
if __name__ == "__main__":
	if(algorithm == "SANI"):
		if(trainMultipleFiles):
			train(trainMultipleFiles=trainMultipleFiles)
		else:
			trainMinimal()
	else:
		print("main error: algorithm == unknown")
		
