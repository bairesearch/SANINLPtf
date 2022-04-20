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


#if(useLearningRuleBackpropagation):
def generatePrediction(ZlastLayer, Whead):
	#print("ZlastLayer.shape = ", ZlastLayer.shape) 
	#print("Whead.shape = ", Whead.shape) 
	pred = tf.matmul(ZlastLayer, Whead)
	pred = tf.nn.softmax(pred)	#tf.nn.sigmoid(ZlastLayer)
	#print("pred.shape = ", pred.shape) 
	return pred				
	#if POStagSequence, pred is compared against single target/POS value (as Ydataset1PartSmall0000.dat is defined as having a single class target: unambiguous pos prediction)
	#if POStagSentence, pred is compared against multiple targets/POS values
		
def defineTrainingParametersSANI(dataset, trainMultipleFiles):
	
	#Training parameters
	if((algorithmSANI == "sharedModulesBinary") and (ANNtf2_globalDefs.testHarness)):	
		learningRate = 0.001
		trainingSteps = 1
		numEpochs = 1
	else:
		if(trainMultipleFiles):
			learningRate = 0.001
			if(dataset == "POStagSentence"):
				trainingSteps = 10000
			elif(dataset == "POStagSequence"):
				trainingSteps = 10000
			elif(dataset == "SmallDataset"):
				trainingSteps = 1000
			numEpochs = 10
		else:
			learningRate = 0.001
			if(dataset == "POStagSentence"):
				trainingSteps = 10000
			elif(dataset == "POStagSequence"):
				trainingSteps = 10000
			elif(dataset == "SmallDataset"):
				trainingSteps = 1000
			numEpochs = 1

	if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
		batchSize = 1	#4	#32	#128	#256	#1 is required for hebbian learning
		displayStep = 1	
	elif(algorithmSANI == "sharedModulesBinary"):
		batchSize = 1	#4	#32	#128	#256
		displayStep = 1	
	else:
		if(allowMultipleSubinputsPerSequentialInput):
			batchSize = 100
			displayStep = 100
		else:
			batchSize = 10	#50
			displayStep = 100	

	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
			
			
def defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, useSmallSentenceLengths, numberOfFeaturesPerWord):
	
	#useSmallSentenceLengths not implemented

	if(inputNumberFeaturesForCurrentWordOnly):
		inputLength = numberOfFeaturesPerWord
	else:
		inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen
	firstLayerSize = inputLength*inputLength	#or datasetNumFeatures
	#if(supportFeedback):
	#	layerSizeMultiplier = 2
	#else:
	#	layerSizeMultiplier = 1

	if(dataset == "POStagSentence"):
		n_x = inputLength #datasetNumFeatures
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
				n_h_2 = int(firstLayerSize*inputLength)
				n_h_3 = int(firstLayerSize*inputLength*inputLength)
				n_h_4 = int(firstLayerSize*inputLength*inputLength)
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
			
	elif(dataset == "POStagSequence"):		
		n_x = num_input_neurons #datasetNumFeatures
		n_y = num_output_neurons  #datasetNumClasses
		n_h_0 = n_x

		if(layerSizeConvergence):
			n_h_1 = int(firstLayerSize)
			n_h_2 = int(firstLayerSize/2)
			n_h_3 = int(firstLayerSize/4)
			n_h_4 = int(firstLayerSize/8)
			#n_h_1 = int(firstLayerSize*3)
			#n_h_2 = int(firstLayerSize/2)
		else:
			n_h_1 = int(firstLayerSize*10)
			n_h_2 = int(firstLayerSize*10)
			n_h_3 = int(firstLayerSize*10)
			n_h_4 = int(firstLayerSize*10)
			
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
		print("dataset unsupported")
		exit()
	
	numberOfLayers = len(n_h)-1
	
	for l1 in range(0, numberOfLayers+1):
		print("n_h[", l1, "] = ", n_h[l1]) 
		
	
	return n_h, numberOfLayers


def defineNeuralNetworkParametersSANI(n_h, numberOfLayers, Cseq, CseqLayer, n_h_cumulative, WRseq, WR, BR, Wseq, Bseq, W, B):

	randomNormal = tf.initializers.RandomNormal()
			
	if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
		if(supportFullConnectivity):
			for l1 in range(1, numberOfLayers+1):
				if(supportFeedback):
					l2Max = numberOfLayers
				else:
					l2Max = l1-1
				for l2 in range(0, l2Max+1):

					#print("\tl = " + str(l1))
					#print("\tl2 = " + str(l2))
					#print("\tn_h[l1] = " + str(n_h[l1]))
					for s in range(numberOfSequentialInputs):
						#print("\t\ts = " + str(s))
						if(useFullConnectivitySparsity):
							sparsityLevel = getSparsityLevel(l1,l2,n_h)
							WseqNP = np.random.exponential(scale=sparsityLevel, size=(n_h[l2], n_h[l1]))
						else:
							WseqNP = np.random.rand(n_h[l2], n_h[l1])	#uniform distribution
						Wseq[generateParameterNameSeqSkipLayers(l1, l2, s, "Wseq")] = tf.Variable(WseqNP, dtype=tf.float32)
						if(l2 == 0):
							Bseq[generateParameterNameSeq(l1, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l1]), dtype=tf.float32)		#not currently trained
													
						if(recordNetworkWeights):
							if(recordSubInputsWeighted):
								WRseq[generateParameterNameSeqSkipLayers(l1, l2, s, "Wseq")] = tf.Variable(tf.zeros([n_h[l2], n_h[l1]]), dtype=tf.float32)	
								
				if(performSummationOfSequentialInputsWeighted):	
					W[generateParameterName(l1, "W")] = tf.Variable(randomNormal([numberOfSequentialInputs, n_h[l1]], dtype=tf.float32))	#randomNormal	#note the architecture of this weight matrix is different than a normal weight matrix. Every set of numberOfSequentialInputs (e.g. 3) represents a unique set of sequential neuron inputs (artificial neurons), and are mapped to an independent neuron (real neuron)
					B[generateParameterName(l1, "B")] = tf.Variable(tf.zeros(n_h[l1]), tf.float32)

					if(recordNetworkWeights):
						if(recordSequentialInputsWeighted):	
							WR[generateParameterNameSkipLayers(l1, "WR")] = tf.Variable(tf.zeros([numberOfSequentialInputs, n_h[l1]]), dtype=tf.float32)
						if(recordNeuronsWeighted):		
							BR[generateParameterName(l1, "BR")] = tf.Variable(tf.zeros(n_h[l1]), tf.float32)				
		else:
			print("defineNeuralNetworkParametersSANI error: sharedModulesNonContiguousFullConnectivity requires supportFullConnectivity")
	else:
		if((algorithmSANI == "sharedModulesBinary") and (ANNtf2_globalDefs.testHarness)):
			if(supportSkipLayers):
				# simple net:
				#
				# L0: w0:1 0 [w1:1] 0 [w2:1] 0 0 0 0 0 . . . 0 0	features [53]
				#     |   /         /
				# L1: x  x      /
				#     |    /
				# L2: x  x

				numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInput(0)

				CseqNPl1c0 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
				CseqNPl1c1 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
				CseqNPl2c0 = np.zeros((numberSubinputsPerSequentialInput, n_h[2]))
				CseqNPl2c1 = np.zeros((numberSubinputsPerSequentialInput, n_h[2]))
				CseqLayerNPl1c0 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
				CseqLayerNPl1c1 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
				CseqLayerNPl2c0 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
				CseqLayerNPl2c1 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))

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
			if(supportSkipLayers):
				n_h_cumulativeNP = np.zeros((numberOfLayers+2), dtype=int)
				n_h_cumulativeNP[0] = 0	#first row always set to 0 for indexing purposes

			for l in range(1, numberOfLayers+1):
				#print("\tl = " + str(l))
				for s in range(numberOfSequentialInputs):
					#print("\t\ts = " + str(s))
					if(useSparseTensors):
						if(allowMultipleSubinputsPerSequentialInput):

							numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInput(s)

							if(supportSkipLayers):
								#neuronIndex = np.random.randint(0, n_h_cumulativeNP[l]+1, n_h[l])
								CseqNP = np.zeros((numberSubinputsPerSequentialInput, n_h[l]))
								if(supportFeedback):
									print("defineNeuralNetworkParametersSANI error: supportFeedback requires supportFullConnectivity")
									CseqLayerNPmax = numberOfLayers
								else:
									CseqLayerNPmax = l-1
								CseqLayerNP = np.random.randint(0, CseqLayerNPmax+1, (numberSubinputsPerSequentialInput, n_h[l]))	#this can be modified to make local/distant connections more probable
								for i in range(numberSubinputsPerSequentialInput):
									for j in range(n_h[l]):
										l2 = CseqLayerNP[i, j]
										CseqNP[i,j] = np.random.randint(0, n_h[l2], 1)
								Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
								CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")] = tf.Variable(CseqLayerNP, dtype=tf.int32)
							else:
								CseqNP = np.random.randint(0, n_h[l-1]+1, (numberSubinputsPerSequentialInput, n_h[l]))	#note +1 is required because np.random.randint generates int between min and max-1
								Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)

							if(performSummationOfSubInputsWeighted):
								WseqNP = np.random.rand(numberSubinputsPerSequentialInput, n_h[l])
								Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(WseqNP, dtype=tf.float32)
								Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)
	
							if(recordNetworkWeights):
								if(recordSubInputsWeighted):
									WRseqNP = np.random.rand(numberSubinputsPerSequentialInput, n_h[l])
									WRseq[generateParameterNameSeq(l, s, "WRseq")] = tf.Variable(WRseqNP, dtype=tf.float32)						

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
								CseqNP = np.random.randint(0, n_h[l-1]+1, n_h[l])	#note +1 is required because np.random.randint generates int between min and max-1
								Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
					else:
						if(performSummationOfSubInputsWeighted):
							if(supportSkipLayers):
								Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(randomNormal([n_h_cumulativeNP[l], n_h[l]], dtype=tf.float32))	#older supportSkipLayers implementation uses n_h_cumulativeNP (should be upgraded)
								Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)
							else:
								Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]], dtype=tf.float32))
								Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)

				if(performSummationOfSequentialInputsWeighted):	
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


def getSparsityLevel(l, l2, n_h):
	if(useFullConnectivitySparsity):
		calibrationValue = 1.0 #exponential scale calibration factor - requires calibration
		averageNumberNeuronsActivePerLayer = 1.0
		sparsityLevel = averageNumberNeuronsActivePerLayer*calibrationValue	#assume ~1 input active per layer

		#averageLayerActivationLevel = 1.0/(n_h[l]*n_h[l2]*numberOfSequentialInputs) * neuronActivationFiringThreshold
		#sparsityLevel = averageLayerActivationLevel
		sparsityLevel = 1.0
	else:
		sparsityLevel = 1.0	#probability of initial strong neural connection per neuron in layer
	
	return sparsityLevel


def calculateNumberSubinputsPerSequentialInput(s):

	if(allowMultipleSubinputsPerSequentialInput):
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
		#calculateNumberSubinputsPerSequentialInput function should not have been called
		numberSubinputsPerSequentialInput = 1	
		
	return numberSubinputsPerSequentialInput
	
