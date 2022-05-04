"""SANItf2.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and Tensorflow 2.1+ 

conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
	
# Usage:
python3 SANItf2.py

# Description:
SANItf - train a Sequentially Activated Neuronal Input neural network (SANI)

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
import SANItf2_algorithmSANIglobalDefs

if(SANItf2_algorithmSANIglobalDefs.printGraphVisualisation):
	#tf.config.run_functions_eagerly(True)	#not supported by tensorflow graph construction
	tf.get_logger().setLevel('ERROR')	#WARNING:tensorflow:From /.../python3.7/site-packages/tensorflow/python/ops/summary_ops_v2.py:1203: start (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01
	graphVisualisationFolder = 'summaries'
	
#select algorithm:
algorithm = "SANI"	#sequentially activated neuronal input artificial neural network	#incomplete+non-convergent

suppressGradientDoNotExistForVariablesWarnings = False

if(SANItf2_algorithmSANIglobalDefs.useLearningRuleBackpropagation):
	costCrossEntropyWithLogits = False
else:
	costCrossEntropyWithLogits = True	#binary categorisation

if(algorithm == "SANI"):
	#set algorithmSANI in SANItf2_algorithmSANIglobalDefs
	if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
		import SANItf2_algorithmSANIsharedModulesNonContiguousFullConnectivity as SANItf2_algorithm
		#if(not useHebbianLearningRule):	#no optimisation/cost function used
	elif(algorithmSANI == "sharedModulesBinary"):
		import SANItf2_algorithmSANIsharedModulesBinary as SANItf2_algorithm
	elif(algorithmSANI == "sharedModules"):
		import SANItf2_algorithmSANIsharedModules as SANItf2_algorithm
	elif(algorithmSANI == "repeatedModules"):
		import SANItf2_algorithmSANIrepeatedModules as SANItf2_algorithm
	
						
#learningRate, trainingSteps, batchSize, displayStep, numEpochs = -1

#performance enhancements for development environment only:
trainMultipleFiles = False	#def:True	#switch increases performance during development	#eg data-POStagSentence
trainMultipleNetworks = False	#trial improve classification accuracy by averaging over multiple independently trained networks (test)
numberOfNetworks = 1

if(trainMultipleFiles):
	fileIndexFirst = 0
	if(SANItf2_algorithmSANIglobalDefs.debugUseSmallSentenceLengths):
		fileIndexLast = 11
	else:
		fileIndexLast = 1202
				
#loadDatasetType3 parameters:
#if generatePOSunambiguousInput=True, generate POS unambiguous permutations for every POS ambiguous data example/experience
#if onlyAddPOSunambiguousInputToTrain=True, do not train network with ambiguous POS possibilities
#if generatePOSunambiguousInput=False and onlyAddPOSunambiguousInputToTrain=False, requires simultaneous propagation of different (ambiguous) POS possibilities

limitSentenceLengths = True	#mandatory
if(limitSentenceLengths):
	if(SANItf2_algorithmSANIglobalDefs.debugUseSmallSentenceLengths): 
		limitSentenceLengthsSize = 10	#or 30
	else:
		limitSentenceLengthsSize = 100	#or 50
	
if(algorithm == "SANI"):
	dataset = SANItf2_algorithm.dataset
	if(dataset == "POStagSentence"):
		numberOfFeaturesPerWord = -1
		paddingTagIndex = -1
		if(SANItf2_algorithmSANIglobalDefs.useLearningRuleBackpropagation):
			generateSequenceNextWordPredictionTargets = True
		if(SANItf2_algorithm.algorithmSANI == "sharedModules"):
			if(SANItf2_algorithm.allowMultipleContributingSubinputsPerSequentialInput):
				generatePOSunambiguousInput = False
				onlyAddPOSunambiguousInputToTrain = False
			else:
				generatePOSunambiguousInput = False
				onlyAddPOSunambiguousInputToTrain = False	#OLD: True				
		else:
			generatePOSunambiguousInput = False
			onlyAddPOSunambiguousInputToTrain = False
	elif(dataset == "POStagSequence"):
		addOnlyPriorUnidirectionalPOSinputToTrain = True
		generateSequenceNextWordPredictionTargets = False	#assume batchY is already pos unambiguous
		trainDataIncludesSentenceOutOfBoundsIndex = True	#Ydataset1PartSmall0000.dat:52 classes, Xdataset1PartSmall0000.dat:53 features per word (train data includes POS_INDEX_OUT_OF_SENTENCE_BOUNDS [53rd index] if !GIA_PREPROCESSOR_POS_TAGGER_DATABASE_DO_NOT_TRAIN_POS_INDEX_OUT_OF_SENTENCE_BOUNDS)
	elif(dataset == "wikiXmlDataset"):
		#should be defined as preprocessor defs (non-variable);
		if(SANItf2_algorithmSANIglobalDefs.useLearningRuleBackpropagation):
			generateSequenceNextWordPredictionTargets = True
		NLPsequentialInputTypeTrainWordVectors = False	#False mandatory: lookup word vectors from preexisting database (do not train them)


			
if(SANItf2_algorithmSANIglobalDefs.debugUseSmallSequenceDataset):
	dataset1FileNameXstart = "Xdataset1PartSmall"
	dataset1FileNameYstart = "Ydataset1PartSmall"
	dataset3FileNameXstart = "Xdataset3PartSmall"
	dataset4FileNameXstart = "Xdataset4PartSmall"
else:
	dataset1FileNameXstart = "Xdataset1Part"
	dataset1FileNameYstart = "Ydataset1Part"
	dataset3FileNameXstart = "Xdataset3Part"
	dataset4FileNameXstart = "Xdataset4Part"	
datasetFileNameXend = ".dat"
datasetFileNameYend = ".dat"
datasetFileNameStart = "datasetPart"
datasetFileNameEnd = ".dat"
xmlDatasetFileNameEnd = ".xml"



def defineTrainingParameters(dataset, numberOfFeaturesPerWord=None, paddingTagIndex=None):
	SANItf2_algorithm.defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWord, paddingTagIndex)
	return SANItf2_algorithm.defineTrainingParametersSANIwrapper(dataset, trainMultipleFiles)

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, debugUseSmallSentenceLengths=None, numberOfFeaturesPerWord=None):
	return SANItf2_algorithm.defineNetworkParametersSANIwrapper(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, debugUseSmallSentenceLengths, numberOfFeaturesPerWord)

def defineNeuralNetworkParameters():
	return SANItf2_algorithm.defineNeuralNetworkParameters()

#define default forward prop function for test (identical to below);
def neuralNetworkPropagationTest(test_x, networkIndex=1):
	return SANItf2_algorithm.neuralNetworkPropagation(test_x, networkIndex)

#define default forward prop function for backprop weights optimisation;
def neuralNetworkPropagation(x, networkIndex=1, l=None):
	return SANItf2_algorithm.neuralNetworkPropagation(x, networkIndex)
	
#if(SANItf2_algorithmSANIglobalDefs.printGraphVisualisation):
@tf.function
def neuralNetworkPropagationGraph(x, networkIndex=1, l=None):
	return SANItf2_algorithm.neuralNetworkPropagation(x, networkIndex)


summary_writer = None

def trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display):

	if(SANItf2_algorithmSANIglobalDefs.useLearningRuleBackpropagation):
		#untested;
		if(generateSequenceNextWordPredictionTargets):
			#generate batchY based on nextWord in sequence
			num_input_neurons = batchX.shape[1]
			numberOfWordsInConvolutionalWindowSeen = SANItf2_algorithmSANIglobalDefs.getNumberOfWordsInConvolutionalWindowSeenFromDatasetPOStagSequence(dataset, num_input_neurons, numberOfFeaturesPerWord)
			maximumSentenceWord = num_input_neurons//numberOfFeaturesPerWord - numberOfWordsInConvolutionalWindowSeen
			for w in range(0, maximumSentenceWord):
				if(SANItf2_algorithmSANIglobalDefs.SANIsharedModules):
					firstWordIndex = 0
				else:
					firstWordIndex = w*numberOfFeaturesPerWord
				nextWordFeatureIndex = (w+numberOfWordsInConvolutionalWindowSeen)*numberOfFeaturesPerWord				
				batchXsubset = batchX[:, firstWordIndex:nextWordFeatureIndex]
				batchYsubset = batchX[:, nextWordFeatureIndex:nextWordFeatureIndex+numberOfFeaturesPerWord]					
				print("batchXsubset.shape = ", batchXsubset.shape)
				print("batchYsubset.shape = ", batchYsubset.shape)
				if(SANItf2_algorithmSANIglobalDefs.printIO):
					print("batchXsubset = ", batchXsubset)
					print("batchYsubset = ", batchYsubset)
									
				loss, acc = executeOptimisation(batchIndex, batchXsubset, batchYsubset, datasetNumClasses, numberOfLayers, optimizer, networkIndex)

				if(display):
					print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
		else:
			loss, acc = executeOptimisation(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)	
			if(display):
				print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
	else:
		#learning algorithm not yet implemented:
		#if(batchSize > 1):
		pred = neuralNetworkPropagation(batchX)	
		print("pred = ", pred)	


def executeOptimisation(batchIndex, x, y, datasetNumClasses, numberOfLayers, optimizer, networkIndex=1):

	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLoss(batchIndex, x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex)

	Wlist = []
	Blist = []
	Wseqlist = []
	Bseqlist = []
	WheadList = []
	WheadList.append(SANItf2_algorithm.Whead)
	for l1 in range(1, numberOfLayers+1):
		if(SANItf2_algorithmSANIglobalDefs.performSummationOfSequentialInputsWeighted):
			Wlist.append(SANItf2_algorithm.W[generateParameterName(l1, "W")])
			Blist.append(SANItf2_algorithm.B[generateParameterName(l1, "B")])
		if(SANItf2_algorithmSANIglobalDefs.performSummationOfSubInputsWeighted):
			if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
				if(SANItf2_algorithmSANIglobalDefs.supportFeedback):
					l2Max = numberOfLayers
				else:
					l2Max = l1-1
				for l2 in range(0, l2Max+1):
					for s in range(SANItf2_algorithmSANIglobalDefs.numberOfSequentialInputs):
						Wseqlist.append(SANItf2_algorithm.Wseq[generateParameterNameSeqSkipLayers(l1, l2, s, "Wseq")])
						if(l2 == 0):
							Bseqlist.append(SANItf2_algorithm.Bseq[generateParameterNameSeq(l1, s, "Bseq")])	
			else:
				for s in range(SANItf2_algorithmSANIglobalDefs.numberOfSequentialInputs):
					Bseqlist.append(SANItf2_algorithm.Bseq[generateParameterNameSeq(l1, s, "Bseq")])
					Wseqlist.append(SANItf2_algorithm.Wseq[generateParameterNameSeq(l1, s, "Wseq")])
				
	trainableVariables = Wlist + Blist + Wseqlist + Bseqlist + WheadList

	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))

	return loss, acc
	
		
def calculatePropagationLoss(batchIndex, x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex=1):
	acc = 0	#only valid for softmax class targets 
	
	if(SANItf2_algorithmSANIglobalDefs.printGraphVisualisation):
		pred = neuralNetworkPropagationGraphPrint(batchIndex, x, networkIndex)	
	else:
		pred = neuralNetworkPropagation(x, networkIndex)
				
	target = y 
	#print("pred = ", pred.shape) 
	#print("target = ", target.shape) 
		
	singleTarget = False
	if(dataset == "POStagSentence"):
		oneHotEncoded = True
		singleTarget = False
		#if(onlyAddPOSunambiguousInputToTrain):
		#	singleTarget = True
		#print("oneHotEncoded")
		target = tf.dtypes.cast(target, tf.float32)
	elif(dataset == "POStagSequence"):
		#print("POStagSequence")
		oneHotEncoded = False
		singleTarget = True
		target = tf.dtypes.cast(target, tf.int32)
	elif(dataset == "wikiXmlDataset"):
		oneHotEncoded = True
		singleTarget = False
		target = tf.dtypes.cast(target, tf.float32)
				
	loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits, oneHotEncoded=oneHotEncoded)
	
	if(singleTarget):
		acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 
		
	#print("x = ", x)
	#print("y = ", y)
	#print("2 loss = ", loss)
	#print("2 acc = ", acc)
	
	return loss, acc
	
	
			
def loadDataset(fileIndex):

	global numberOfFeaturesPerWord
	global paddingTagIndex
	
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
	elif(dataset == "wikiXmlDataset"):
		datasetType4FileName = dataset4FileNameXstart + fileIndexStr + xmlDatasetFileNameEnd

	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY, addOnlyPriorUnidirectionalPOSinputToTrain)
		if(trainDataIncludesSentenceOutOfBoundsIndex):
			datasetNumClasses = datasetNumClasses + 1
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, limitSentenceLengthsSize)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	elif(dataset == "wikiXmlDataset"):
		articles = ANNtf2_loadDataset.loadDatasetType4(datasetType4FileName, limitSentenceLengths, limitSentenceLengthsSize, NLPsequentialInputTypeTrainWordVectors)
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.convertArticlesTreeToSentencesWordVectors(articles, limitSentenceLengthsSize)

	return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y



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
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, SANItf2_algorithmSANIglobalDefs.debugUseSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		fileIndex = 0
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

		shuffleSize = getShuffleSize(datasetNumExamples, batchSize)
		trainDataIndex = 0

		#print("shuffleSize = ", shuffleSize)
		#print("batchSize = ", batchSize)
		#print("train_x.shape = ", train_x.shape)
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

			
#SANItf does not supportMultipleNetworks
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
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, SANItf2_algorithmSANIglobalDefs.debugUseSmallSentenceLengths, numberOfFeaturesPerWord)
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
		if(randomiseFileIndexParse):
			fileIndexShuffledArray = generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f
				
			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

			shuffleSize = getShuffleSize(datasetNumExamples, batchSize)
			trainDataIndex = 0

			#greedy code;
			for l in range(1, maxLayer+1):
				print("l = ", l)
				trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
				#print("trainData = ", trainData)
				trainDataList = []
				trainDataList.append(trainData)
				trainDataListIterators = []
				for trainData in trainDataList:
					trainDataListIterators.append(iter(trainData))
				testBatchX, testBatchY = generateTFbatch(test_x, test_y, batchSize)
				#testBatchX, testBatchY = (test_x, test_y)

				for batchIndex in range(int(trainingSteps)):
					(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
					batchYactual = batchY
					
					for networkIndex in range(1, maxNetwork+1):
						display = False
						#if(l == maxLayer):	#only print accuracy after training final layer
						if(batchIndex % displayStep == 0):
							display = True	
						trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display)
						
					#trainMultipleNetworks code;
					if(trainMultipleNetworks):
						#train combined network final layer
						trainBatchAllNetworksFinalLayer(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, costCrossEntropyWithLogits, display)

				#trainMultipleNetworks code;
				if(trainMultipleNetworks):
					testBatchAllNetworksFinalLayer(testBatchX, testBatchY, datasetNumClasses, numberOfLayers)
				else:
					pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
					if(greedy):
						print("Test Accuracy: l: %i, %f" % (l, calculateAccuracy(pred, testBatchY)))
					else:
						print("Test Accuracy: %f" % (calculateAccuracy(pred, testBatchY)))


def generateRandomisedIndexArray(indexFirst, indexLast, arraySize=None):
	fileIndexArray = np.arange(indexFirst, indexLast+1, 1)
	#print("fileIndexArray = " + str(fileIndexArray))
	if(arraySize is None):
		np.random.shuffle(fileIndexArray)
		fileIndexRandomArray = fileIndexArray
	else:
		fileIndexRandomArray = random.sample(fileIndexArray.tolist(), arraySize)
	
	print("fileIndexRandomArray = " + str(fileIndexRandomArray))
	return fileIndexRandomArray
							

def getShuffleSize(datasetNumExamples, batchSize):	
	#shuffleSize = min(datasetNumExamples, batchSize)	#heuristic: 10*batchSize
	shuffleSize = datasetNumExamples
	return shuffleSize

#developmental;
def neuralNetworkPropagationGraphPrint(batchIndex, x, networkIndex=1):
	summary_writer = tf.summary.create_file_writer(graphVisualisationFolder)
	tf.summary.trace_on(graph=True, profiler=True)	   #tf.summary.trace_on(graph=True)

	pred = neuralNetworkPropagationGraph(x, networkIndex)
	with summary_writer.as_default():
		tf.summary.trace_export(name="executeOptimisation_trace", step=batchIndex, profiler_outdir=graphVisualisationFolder)

	#summary_writer.flush()
	print("printGraphVisualisation: neuralNetworkPropagationGraphPrint")
	print("execute this: tensorboard --logdir=" + graphVisualisationFolder + " --host localhost --port 8088")
	print("then open http://localhost:8088/ with webbrowser")
	exit()
	
	return pred
					
if __name__ == "__main__":
	if(algorithm == "SANI"):
		if(trainMultipleFiles):
			train(trainMultipleFiles=trainMultipleFiles)
		else:
			trainMinimal()
	else:
		print("main error: algorithm == unknown")
		
