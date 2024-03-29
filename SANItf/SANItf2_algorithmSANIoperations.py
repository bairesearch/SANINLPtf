"""SANItf2_algorithmSANIoperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SANItf2.py

# Usage:
see SANItf2.py

# Description:
SANItf algorithm SANI (Sequentially Activated Neuronal Input) operations

"""

import tensorflow as tf
import numpy as np
import ANNtf2_globalDefs
from SANItf2_algorithmSANIglobalDefs import *
from ANNtf2_operations import * #generateParameterNameSeq, generateParameterName
from numpy import sqrt

#if(useLearningRuleBackpropagation):
def generatePrediction(ZlastLayer, Whead, applySoftmax=True):
	#print("ZlastLayer = ", ZlastLayer) 
	#print("Whead = ", Whead) 
	pred = tf.matmul(ZlastLayer, Whead)
	if(applySoftmax):
		pred = tf.nn.softmax(pred)	#tf.nn.sigmoid(pred)
	#print("pred = ", pred)
	#print("pred.shape = ", pred.shape) 
	return pred				
	#if POStagSequence, pred is compared against single target/POS value (as Ydataset1PartSmall0000.dat is defined as having a single class target: unambiguous pos prediction)
	#if POStagSentence, pred is compared against multiple targets/POS values
		
def defineTrainingParametersSANI(dataset, trainMultipleFiles):
	
	#Training parameters
	if((ANNtf2_globalDefs.testHarness) and (algorithmSANI == "sharedModulesBinary")):	
		learningRate = 0.001
		trainingSteps = 1
		numEpochs = 1
	elif(debugTrainSingleBatch):
	   learningRate = 0.001
	   trainingSteps = 1
	   numEpochs = 1000
	else:
		learningRate = 0.001
		trainingSteps = 10000
		numEpochs = 10
		#if(trainMultipleFiles):
		#else:
	
	if(debugUseSmallBatchSize):
	   batchSize = 1
	   displayStep = 1
	else:
		if(dataset == "wikiXmlDataset"):
			#requires more memory
			batchSize = 10	#default: 10
			displayStep = 1
		else:
			batchSize = 100
			displayStep = 1	

	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
			
			
def defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, debugUseSmallSentenceLengths, numberOfFeaturesPerWord):
	
	#debugUseSmallSentenceLengths not implemented
	global n_h
	
	layerSizeMultiplier = numberOfFeaturesPerWord
	if(debugUseSmallLayers):
		firstLayerSize = numberOfFeaturesPerWord
	else:
		if(dataset == "wikiXmlDataset"):
			firstLayerSize = numberOfFeaturesPerWord*10
		else:
			firstLayerSize = numberOfFeaturesPerWord*layerSizeMultiplier	#or datasetNumFeatures
	#if(supportFeedback):
	#	layerSizeMultiplier2 = 2
	#else:
	#	layerSizeMultiplier2 = 1
	
	if((dataset == "POStagSentence") or (dataset == "POStagSequence") or (dataset == "wikiXmlDataset")):
	
		numberOfWordsInConvolutionalWindowSeen = getNumberOfWordsInConvolutionalWindowSeenFromDatasetPOStagSequence(dataset, num_input_neurons, numberOfFeaturesPerWord)
		
		if(inputNumberFeaturesForCurrentWordOnly):
			inputLength = numberOfFeaturesPerWord
		else:
			inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen
		n_x = inputLength #datasetNumFeatures
		#print("n_x = ", n_x)
		
		if(dataset == "POStagSequence"):
			n_y = num_output_neurons  #typically equivalent to datasetNumClasses and numberOfFeaturesPerWord
		else:
			if(useLearningRuleBackpropagation):
				n_y = numberOfFeaturesPerWord  
			else:
				n_y = 1	#SANIshared uses a single output neuron (either 1 or 0)	#if multiple output classes: n_y = num_output_neurons-1 or datasetNumClasses-1	
										
		n_h_0 = n_x	
		if(layerSizeConvergence):
			#FUTURE: the number of neurons/connections should be greatly increased, then pruned
			n_h_1 = int(firstLayerSize)
			n_h_2 = int(firstLayerSize/2)
			n_h_3 = int(firstLayerSize/4)
			n_h_4 = int(firstLayerSize/8)
			#n_h_1 = int(firstLayerSize*3)
			#n_h_2 = int(firstLayerSize/2)
		else:
			layerSizeDivergenceExponential = False	#else linear	
			if(layerSizeDivergenceExponential):
				n_h_1 = int(firstLayerSize)
				n_h_2 = int(firstLayerSize*layerSizeMultiplier)
				n_h_3 = int(firstLayerSize*layerSizeMultiplier*layerSizeMultiplier)
				n_h_4 = int(firstLayerSize*layerSizeMultiplier*layerSizeMultiplier)
			else:
				#*x for skip layers #FUTURE: upgrade to support multiple permutations
				n_h_1 = int(firstLayerSize*1)
				n_h_2 = int(firstLayerSize*2)
				n_h_3 = int(firstLayerSize*3)
				n_h_4 = int(firstLayerSize*4)

		if(useLearningRuleBackpropagation):
			n_h_5 = n_h_4
		else:		
			n_h_5 = n_y
		n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5]		
	#elif(dataset == "SmallDataset"):
	#	n_h_1 = 4
	#	n_h_2 = 4
	#	if(useLearningRuleBackpropagation):
	#		n_h_3 = n_h_2
	#	else:		
	#		n_h_3 = n_y
	#	n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
	else:
		print("defineNetworkParametersSANI: dataset unsupported")
		exit()
	
	numberOfLayers = len(n_h)-1
	
	for l1 in range(0, numberOfLayers+1):
		print("n_h[", l1, "] = ", n_h[l1]) 
		
	return n_h, numberOfLayers


def defineNeuralNetworkParametersSANI(n_h, numberOfLayers, Cseq, CseqLayer, n_h_cumulative, WRseq, WR, BR, Wseq, Bseq, W, B):

	randomNormal = tf.initializers.RandomNormal()	#mean=0.0, stddev=0.05
	#randomXavier = tf.initializers.GlorotUniform()	#NO: randomNormalHigh; mean=0.0, stddev=0.1
		
	if(not ANNtf2_globalDefs.testHarness):	
		if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
			if(supportFullConnectivity):
				for l1 in range(1, numberOfLayers+1):
					if(supportFeedback):
						l2Max = numberOfLayers
					else:
						l2Max = l1-1
					if(supportSkipLayers):
						l2Min = 0
					else:
						l2Min = l1-1
					for l2 in range(l2Min, l2Max+1):

						#print("\tl = " + str(l1))
						#print("\tl2 = " + str(l2))
						#print("\tn_h[l1] = " + str(n_h[l1]))
						for s in range(numberOfSequentialInputs):
							#print("\t\ts = " + str(s))
							WseqTF = randomNormal([n_h[l2], n_h[l1]], dtype=tf.float32)
							#WseqNP = np.random.normal(size=(n_h[l2], n_h[l1]))	#not equivalent to tensorflow RandomNormal weight initialisation
							if(useFullConnectivitySparsity):
								sparsityLevel = getSparsityLevelFullConnectivity(l1, l2, n_h)
								probabilityUniform = tf.random.uniform(shape=[n_h[l2], n_h[l1]], minval=0.0, maxval=1.0)	#0 to 1
								sparsityPass = tf.less(probabilityUniform, sparsityLevel)
								sparsityPass = tf.dtypes.cast(sparsityPass, tf.float32)
								WseqTF = tf.multiply(WseqTF, sparsityPass)	#zero some weights
								#print("sparsityPass = ", sparsityPass)
								#print("WseqTF = ", WseqTF)
								#print("tf.math.reduce_mean(WseqTF) = ", tf.math.reduce_mean(WseqTF))	#~0
							Wseq[generateParameterNameSeqSkipLayers(l1, l2, s, "Wseq")] = tf.Variable(WseqTF)
							if(l2 == l2Min):
								Bseq[generateParameterNameSeq(l1, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l1]), dtype=tf.float32)		#not currently trained

							if(recordNetworkWeights):
								if(recordSubInputsWeighted):
									WRseq[generateParameterNameSeqSkipLayers(l1, l2, s, "Wseq")] = tf.Variable(tf.zeros([n_h[l2], n_h[l1]]), dtype=tf.float32)	

					if(performFunctionOfSequentialInputsWeighted):	
						W[generateParameterName(l1, "W")] = tf.Variable(randomNormal([numberOfSequentialInputs, n_h[l1]], dtype=tf.float32))	#randomNormal	#note the architecture of this weight matrix is different than a normal weight matrix. Every set of numberOfSequentialInputs (e.g. 3) represents a unique set of sequential neuron inputs (artificial neurons), and are mapped to an independent neuron (real neuron)
						B[generateParameterName(l1, "B")] = tf.Variable(tf.zeros(n_h[l1]), tf.float32)

						if(recordNetworkWeights):
							if(recordSequentialInputsWeighted):	
								WR[generateParameterNameSkipLayers(l1, "WR")] = tf.Variable(tf.zeros([numberOfSequentialInputs, n_h[l1]]), dtype=tf.float32)
							if(recordNeuronsWeighted):		
								BR[generateParameterName(l1, "BR")] = tf.Variable(tf.zeros(n_h[l1]), tf.float32)				
			else:
				print("defineNeuralNetworkParametersSANI error: sharedModulesNonContiguousFullConnectivity requires supportFullConnectivity")			
		elif((algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules") or (algorithmSANI == "repeatedModules")):
			if(supportSkipLayers):
				n_h_cumulativeNP = np.zeros((numberOfLayers+2), dtype=int)
				n_h_cumulativeNP[0] = 0	#first row always set to 0 for indexing purposes

			for l in range(1, numberOfLayers+1):
				#print("\tl = " + str(l))
				for s in range(numberOfSequentialInputs):
					#print("\t\ts = " + str(s))
					if(useSparseTensors):
						if(allowMultipleSubinputsPerSequentialInput):
							numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)

							if(supportSkipLayers):
								#neuronIndex = np.random.randint(0, n_h_cumulativeNP[l]+1, n_h[l])
								CseqNP = np.zeros((numberSubinputsPerSequentialInput, n_h[l]))
								if(supportFeedback):
									print("defineNeuralNetworkParametersSANI error: supportFeedback requires supportFullConnectivity")
									CseqLayerNPmax = numberOfLayers
								else:
									CseqLayerNPmax = l-1
								CseqLayerNP = np.random.randint(0, CseqLayerNPmax+1, (numberSubinputsPerSequentialInput, n_h[l]))	#this can be modified to make local/distant connections more probable	#note +1 is required because np.random.randint generates int between min and max-1
								for i in range(numberSubinputsPerSequentialInput):
									for j in range(n_h[l]):
										l2 = CseqLayerNP[i, j]
										CseqNP[i,j] = np.random.randint(0, n_h[l2], 1)
								Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
								CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")] = tf.Variable(CseqLayerNP, dtype=tf.int32)
							else:
								#print("n_h[l-1] = ", n_h[l-1])
								CseqNP = np.random.randint(0, n_h[l-1], (numberSubinputsPerSequentialInput, n_h[l]))
								Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)

							if(performFunctionOfSubInputsWeighted):
								#Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(randomXavier([numberSubinputsPerSequentialInput, n_h[l]], dtype=tf.float32))
								n = numberSubinputsPerSequentialInput
								Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(tf.random.uniform(shape=[numberSubinputsPerSequentialInput, n_h[l]], minval=-(1/sqrt(n)), maxval= 1/sqrt(n)))
								#print("Wseq = ", Wseq[generateParameterNameSeq(l, s, "Wseq")])
								Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)

							if(recordNetworkWeights):
								if(recordSubInputsWeighted):
									WRseq[generateParameterNameSeq(l, s, "WRseq")] = tf.Variable(randomNormal([numberSubinputsPerSequentialInput, n_h[l]], dtype=tf.float32))						

						else:
							if(supportSkipLayers):
								#neuronIndex = np.random.randint(0, n_h_cumulativeNP[l]+1, n_h[l])
								CseqNP = np.zeros((n_h[l]))
								CseqLayerNP = np.random.randint(0, l, n_h[l])	#this can be modified to make local/distant connections more probable
								for index, l2 in enumerate(CseqLayerNP):
									CseqNP[index] = np.random.randint(0, n_h[l2], 1)
								Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
								CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")] = tf.Variable(CseqLayerNP, dtype=tf.int32)
							else:
								CseqNP = np.random.randint(0, n_h[l-1], n_h[l])
								Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
					else:
						if(performFunctionOfSubInputsWeighted):
							if(supportSkipLayers):
								Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(randomNormal([n_h_cumulativeNP[l], n_h[l]], dtype=tf.float32))	#older supportSkipLayers implementation uses n_h_cumulativeNP (should be upgraded)
								Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)
							else:
								Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]], dtype=tf.float32))
								Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)

				if(performFunctionOfSequentialInputsWeighted):	
					W[generateParameterName(l, "W")] = tf.Variable(randomNormal([numberOfSequentialInputs, n_h[l]], dtype=tf.float32))	#randomNormal	#note the architecture of this weight matrix is different than a normal weight matrix. Every set of numberOfSequentialInputs (e.g. 3) represents a unique set of sequential neuron inputs (artificial neurons), and are mapped to an independent neuron (real neuron)
					B[generateParameterName(l, "B")] = tf.Variable(tf.zeros(n_h[l]), tf.float32)

				if(recordNetworkWeights):
					if(recordSequentialInputsWeighted):	
						WR[generateParameterName(l, "WR")] = tf.Variable(randomNormal([numberOfSequentialInputs, n_h[l]], dtype=tf.float32))	#randomNormal	#note the architecture of this weight matrix is different than a normal weight matrix. Every set of numberOfSequentialInputs (e.g. 3) represents a unique set of sequential neuron inputs (artificial neurons), and are mapped to an independent neuron (real neuron)
					if(recordNeuronsWeighted):		
						BR[generateParameterName(l, "BR")] = tf.Variable(tf.zeros(n_h[l]), tf.float32)

				if(supportSkipLayers):
					n_h_cumulativeNP[l] = n_h_cumulativeNP[l-1] + n_h[l-1]

			if(supportSkipLayers):
				n_h_cumulativeNP[numberOfLayers+1] = n_h_cumulativeNP[numberOfLayers] + n_h[numberOfLayers]	#not used
				n_h_cumulative['n_h_cumulative'] = tf.Variable(n_h_cumulativeNP, dtype=tf.int32)
	else:	#ANNtf2_globalDefs.testHarness:
		if(algorithmSANI == "sharedModulesBinary"): 
			if(supportSkipLayers):
				# simple net:
				#
				# L0: w0:1 0 [w1:1] 0 [w2:1] 0 0 0 0 0 . . . 0 0	features [53]
				#     |   /         /
				# L1: x  x      /
				#     |    /
				# L2: x  x

				numberSubinputsPerSequentialInput1 = calculateNumberSubinputsPerSequentialInputSparseTensors(1, 0)
				CseqNPl1c0 = np.zeros((numberSubinputsPerSequentialInput1, n_h[1]))
				CseqNPl1c1 = np.zeros((numberSubinputsPerSequentialInput1, n_h[1]))
				numberSubinputsPerSequentialInput2 = calculateNumberSubinputsPerSequentialInputSparseTensors(2, 0)
				CseqNPl2c0 = np.zeros((numberSubinputsPerSequentialInput2, n_h[2]))
				CseqNPl2c1 = np.zeros((numberSubinputsPerSequentialInput2, n_h[2]))
				CseqLayerNPl1c0 = np.zeros((numberSubinputsPerSequentialInput1, n_h[1]))
				CseqLayerNPl1c1 = np.zeros((numberSubinputsPerSequentialInput1, n_h[1]))
				CseqLayerNPl2c0 = np.zeros((numberSubinputsPerSequentialInput2, n_h[1]))
				CseqLayerNPl2c1 = np.zeros((numberSubinputsPerSequentialInput2, n_h[1]))

				CseqNPl1c0[0, 0] = 0
				CseqNPl1c1[0, 0] = 1
				CseqLayerNPl1c0[0, 0] = 0
				CseqLayerNPl1c1[0, 0] = 0
				n_h_cumulativeNP[1] = n_h_cumulativeNP[0] + n_h[0]

				CseqNPl2c0[0, 0] = 0
				CseqNPl2c1[0, 0] = 2
				CseqLayerNPl2c0[0, 0] = 1
				CseqLayerNPl2c1[0, 0] = 0
				n_h_cumulativeNP[2] = n_h_cumulativeNP[1] + n_h[1]

				Cseq[generateParameterNameSeq(1, 0, "Cseq")] = tf.Variable(CseqNPl1c0, dtype=tf.int32)
				Cseq[generateParameterNameSeq(1, 1, "Cseq")] = tf.Variable(CseqNPl1c1, dtype=tf.int32)
				Cseq[generateParameterNameSeq(2, 0, "Cseq")] = tf.Variable(CseqNPl2c0, dtype=tf.int32)
				Cseq[generateParameterNameSeq(2, 1, "Cseq")] = tf.Variable(CseqNPl2c1, dtype=tf.int32)
				CseqLayer[generateParameterNameSeq(1, 0, "CseqLayer")] = tf.Variable(CseqLayerNPl1c0, dtype=tf.int32)
				CseqLayer[generateParameterNameSeq(1, 1, "CseqLayer")] = tf.Variable(CseqLayerNPl1c1, dtype=tf.int32)
				CseqLayer[generateParameterNameSeq(2, 0, "CseqLayer")] = tf.Variable(CseqLayerNPl2c0, dtype=tf.int32)
				CseqLayer[generateParameterNameSeq(2, 1, "CseqLayer")] = tf.Variable(CseqLayerNPl2c1, dtype=tf.int32)	
			else:
				print("error: (algorithmSANI == sharedModulesBinary) and (ANNtf2_globalDefs.testHarness)) requires supportSkipLayers")
				exit()
		else:
			print("error: (ANNtf2_globalDefs.testHarness)) requires algorithmSANI=sharedModulesBinary")
			exit()	

	
def getSparsityLevelFullConnectivity(l, l2, n_h):
	if(useFullConnectivitySparsity):
		if(firstLayerFullConnectivity):
			if(l2 == 0):	#if((l == 1) and (l2 == 0))
				sparsityLevel = 1.0
			else:
				sparsityLevel = probabilityOfActiveConnection
		else:
			sparsityLevel = probabilityOfActiveConnection
	else:
		sparsityLevel = 1.0	#probability of initial strong neural connection per neuron in layer
	
	return sparsityLevel


def calculateNumberSubinputsPerSequentialInputSparseTensors(l, s):

	if(allowMultipleSubinputsPerSequentialInput):
		if((l == 1) and firstLayerFullConnectivity):
			numberSubinputsPerSequentialInput = n_h[0]	#number of features in first layer
		else:
			if(oneSequentialInputHasOnlyOneSubinput):
				if(firstSequentialInputHasOnlyOneSubinput and s==0):
					numberSubinputsPerSequentialInput = 1
				elif(lastSequentialInputHasOnlyOneSubinput and s==numberOfSequentialInputs-1):
					numberSubinputsPerSequentialInput = 1
				else:
					numberSubinputsPerSequentialInput = maxNumberSubinputsPerSequentialInput
			else:
				numberSubinputsPerSequentialInput = maxNumberSubinputsPerSequentialInput
	else:
		#calculateNumberSubinputsPerSequentialInputSparseTensors function should not have been called
		print("calculateNumberSubinputsPerSequentialInputSparseTensors error: !allowMultipleSubinputsPerSequentialInput")
		exit()
		numberSubinputsPerSequentialInput = 1	
		
	return numberSubinputsPerSequentialInput
	

def updateTensorCells(tensorOrig, tensorUpdatesNonZero):

	#print("tensorOrig = ", tensorOrig)
	#print("tensorUpdatesNonZero = ", tensorUpdatesNonZero)

	indices = tf.where(tensorUpdatesNonZero)
	#print("indices = ", indices)
	data = tf.gather_nd(tensorUpdatesNonZero, indices)
	#print("data = ", data)
	
	tensorMod = tf.tensor_scatter_nd_update(tensorOrig, indices, data)
	#print("tensorMod = ", tensorMod)
	
	return tensorMod
