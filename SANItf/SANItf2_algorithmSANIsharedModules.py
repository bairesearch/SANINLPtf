"""SANItf2_algorithmSANIsharedModules.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SANItf2.py

# Usage:
see SANItf2.py

# Description:

SANItf algorithm SANI shared modules - define Sequentially Activated Neuronal Input neural network with shared modules

Neural modules can be shared between different areas of input sequence, eg sentence (cf RNN).
This code mirrors that of GIA Sequence Grammar ANN.
Can parse (by default expects to parse) full sentences; ie features for each word in sentence.

"""

#start common SANItf2_algorithmSANI.py code:

import tensorflow as tf
import numpy as np
from ANNtf2_operations import * #generateParameterNameSeq, generateParameterName
import SANItf2_algorithmSANIoperations
from SANItf2_algorithmSANIglobalDefs import *
import ANNtf2_globalDefs


#parameters
#static parameters (convert to tf.constant?):
Cseq = {}
CseqLayer = {}	
n_h_cumulative = {}
#variable parameters:
WRseq = {}	#weights matrix
WR = {}	#weights matrix
BR = {}	#biases vector
Wseq = {}	#weights matrix
Bseq = {}	#biases vector
W = {}	#weights matrix
B = {}	#biases vector
if(useLearningRuleBackpropagation):
	Whead = [] #final linear layer weights matrix

#parameters
#static parameters (convert to tf.constant?):
#if(not supportFullConnectivity):
#	if(useSparseTensors):
#		Cseq = {}	#connectivity vector
#		if(supportSkipLayers):	
#			CseqLayer = {}	
#			n_h_cumulative = {}
##variable parameters:	
#if((algorithmSANI == "sharedModulesNonContiguousFullConnectivity") or (algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):
#	if(recordNetworkWeights):
#		if(recordSubInputsWeighted):
#			AseqInputVerified = {}
#			WRseq = {}	#weights matrix
#		if(recordSequentialInputsWeighted):
#			WR = {}	#weights matrix
#		if(recordNeuronsWeighted):
#			BR = {}	#biases vector
#if((algorithmSANI == "sharedModulesNonContiguousFullConnectivity") or (algorithmSANI == "sharedModules") or (algorithmSANI == "repeatedModules")):
#	#variable parameters: 
#	if(allowMultipleSubinputsPerSequentialInput):
#		if(performFunctionOfSubInputsWeighted):
#			Wseq = {}	#weights matrix
#			Bseq = {}	#biases vector
#	if(performFunctionOfSequentialInputsWeighted):
#		W = {}	#weights matrix
#		B = {}	#biases vector
	
			
#Network parameters
n_h = []
numberOfLayers = 0

#if((algorithmSANI == "sharedModulesNonContiguousFullConnectivity") or (algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):	#only code to currently use these variables
numberOfFeaturesPerWord = -1
paddingTagIndex = -1
def defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWordNew, paddingTagIndexNew):
	#if((algorithmSANI == "sharedModulesNonContiguousFullConnectivity") or (algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):	#only code to currently use these variables
	global numberOfFeaturesPerWord
	global paddingTagIndex
	numberOfFeaturesPerWord = numberOfFeaturesPerWordNew
	paddingTagIndex = paddingTagIndexNew

def defineNetworkParametersSANIwrapper(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, debugUseSmallSentenceLengths, numberOfFeaturesPerWord):
	global n_h
	global numberOfLayers
	n_h, numberOfLayers = SANItf2_algorithmSANIoperations.defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, debugUseSmallSentenceLengths, numberOfFeaturesPerWord)
	return numberOfLayers
	
def defineTrainingParametersSANIwrapper(dataset, trainMultipleFiles):
	return SANItf2_algorithmSANIoperations.defineTrainingParametersSANI(dataset, trainMultipleFiles)
	

def defineNeuralNetworkParameters():
	global n_h_cumulative
	if(useLearningRuleBackpropagation):
		global Whead
		randomNormal = tf.initializers.RandomNormal()
		Whead = tf.Variable(randomNormal([n_h[numberOfLayers], numberOfFeaturesPerWord], dtype=tf.float32))
	SANItf2_algorithmSANIoperations.defineNeuralNetworkParametersSANI(n_h, numberOfLayers, Cseq, CseqLayer, n_h_cumulative, WRseq, WR, BR, Wseq, Bseq, W, B)
			

#temporary variables for neuralNetworkPropagationSANI:
if(algorithmSANI == "sharedModules"):
	Vseq = {}
	Zseq = {}
	Aseq = {}
	Z = {}
	A = {}
	if(enforceTcontiguityConstraints):
		TMaxSeq = {}
		TMinSeq = {}
		ZseqTadjusted = {}
		TMax = {}
		TMin = {}
	if(resetSequentialInputsIfOnlyFirstInputValid):
		sequentialActivationFound = {}		#records whether last s sequential input was activated
	AseqInputVerified = {}

#end common SANItf2_algorithmSANI.py code



def neuralNetworkPropagation(x, networkIndex=None):
	return neuralNetworkPropagationSANI(x)
	
def neuralNetworkPropagationSANI(x):

	x = tf.dtypes.cast(x, tf.float32)
	
	#note connectivity indexes are used rather than sparse weight matrices due to limitations in current tf2 sparse tensor implementation
	
	#definitions for reference:
	
	#neuron sequential input vars;
	#x/AprevLayer	#output vector (dim: batchSize*n_h[l])
	#if(allowMultipleSubinputsPerSequentialInput):
		#if(useSparseTensors):
			#Cseq	#static connectivity matrix (int) - indexes of neurons on prior layer stored; mapped to W  (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
			#if(supportSkipLayers):
				#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
			#Wseq #weights of connections; see Cseq (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
			#AseqSum	#combination variable
		#else:
			#if(supportSkipLayers):
				#Wseq #weights of connections (dim: n_h_cumulativeNP[l]*n_h[l])
			#else:
				#Wseq #weights of connections (dim: n_h[l-1]*n_h[l])
	#else:
		#Cseq	#static connectivity vector (int) - indexes of neurons on prior layer stored; mapped to W - defines which prior layer neuron a sequential input is connected to (dim: n_h[l])
		#if(supportSkipLayers):
			#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W - defines which prior layer a sequential input is connected to  (dim: n_h[l])
		#Wseq #weights of connections; see Cseq (dim: n_h[l])	
	#Vseq	#mutable verification vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)			#records whether particular neuron sequential inputs are currently active
	#Zseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#Aseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs

	#neuron vars;
	#Q  
	#Z	#neuron activation function input (dim: batchSize*n_h[l])
	#A	#neuron activation function output (dim: batchSize*n_h[l])
	#if(performFunctionOfSequentialInputsWeighted):	
		#W	(dim: numberOfSequentialInputs*n_h[l])	

	#combination vars (per layer);
	#if(performFunctionOfSequentialInputs):
		#these are all used for different methods of sequential input summation:
		#if(performFunctionOfSequentialInputsWeighted):
			#if(performFunctionOfSubInputsNonlinear):
				#AseqWeightedSum	#(dim: batchSize*n_h[l])
			#else:
				#ZseqWeightedSum	#(dim: batchSize*n_h[l])
		#else:
			#if(performFunctionOfSubInputsNonlinear):
				#AseqSum	#(dim: batchSize*n_h[l])
			#else:
				#ZseqSum	#(dim: batchSize*n_h[l])
			
	#Tcontiguity vars;
	#if(enforceTcontiguityConstraints):
		#neuron sequential input vars;
		#TMaxSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a particular sequential input last fired
		#TMinSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a first encapsulated subinput fired
		#if(allowMultipleContributingSubinputsPerSequentialInput):
			#if((resetSequentialInputsTContiguity) and (s == 0)):
				#ZseqTadjusted	#neuron activation function output vector T adjusted (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs*time
		#neuron vars;
		#TMax	#mutable time vector (dim: batchSize*n_h[l]) - same as TMaxSeq[numberOfSequentialInputs-1]
		#TMin	#mutable time vector (dim: batchSize*n_h[l]) - same as TMinSeq[numberOfSequentialInputs-1]
	
	#record vars;
	#if(recordNetworkWeights):
		#neuron sequential input vars;
		#if(recordSubInputsWeighted):
			#AseqInputVerified	#neuron input (sequentially verified) (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l]  - records the subinputs that were used to activate the sequential input)
			#WRseq #weights of connections; see Cseq (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
		#neuron vars;
		#if(recordSequentialInputsWeighted):	
			#WR	(dim: numberOfSequentialInputs*n_h[l])	
		#if(recordNeuronsWeighted):	
			#BR	(dim: n_h[l])
	
	
	batchSize = x.shape[0]
		
	#optimise feed length based on max sentence length in batch:
	#unoptimised: numberOfFeatures = x.shape[1]
	xIsNotPadding = tf.math.not_equal(x, tf.dtypes.cast(paddingTagIndex, tf.float32))
	coordinatesOfNotPadding = tf.where(xIsNotPadding)
	numberOfFeaturesCropped = tf.reduce_max(coordinatesOfNotPadding[:, 1])
	numberOfFeaturesCropped = tf.dtypes.cast(numberOfFeaturesCropped, tf.int32)
	numberOfFeaturesCropped = tf.add(numberOfFeaturesCropped, 1)
	maxNumberOfWordsInSentenceBatch = tf.divide(numberOfFeaturesCropped, numberOfFeaturesPerWord)
	maxNumberOfWordsInSentenceBatch = tf.dtypes.cast(maxNumberOfWordsInSentenceBatch, tf.int32)

	inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen
	
	for l in range(1, numberOfLayers+1):
		Z[generateParameterName(l, "Z")] = tf.zeros([batchSize, n_h[l]])
		A[generateParameterName(l, "A")] = tf.zeros([batchSize, n_h[l]])
		if(enforceTcontiguityConstraints):
			TMax[generateParameterName(l, "TMax")] = tf.zeros([batchSize, n_h[l]], dtype=tf.int32)
			TMin[generateParameterName(l, "TMin")] = tf.zeros([batchSize, n_h[l]], dtype=tf.int32)
		for s in range(numberOfSequentialInputs):
			Vseq[generateParameterNameSeq(l, s, "Vseq")] = tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool)
			Zseq[generateParameterNameSeq(l, s, "Zseq")] = tf.zeros([batchSize, n_h[l]])
			Aseq[generateParameterNameSeq(l, s, "Aseq")] = tf.zeros([batchSize, n_h[l]])
			if(enforceTcontiguityConstraints):
				TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = tf.zeros([batchSize, n_h[l]], dtype=tf.int32)
				TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = tf.zeros([batchSize, n_h[l]], dtype=tf.int32)
				if(allowMultipleContributingSubinputsPerSequentialInput):
					if((resetSequentialInputsTContiguity) and (s == 0)):
						ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")] = tf.zeros([batchSize, n_h[l]])
			if(recordNetworkWeights):
				if(recordSubInputsWeighted):
					numberSubinputsPerSequentialInput = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)
					AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")] = tf.dtypes.cast(tf.zeros([batchSize, numberSubinputsPerSequentialInput, n_h[l]]), dtype=tf.bool)
		
	wMax = maxNumberOfWordsInSentenceBatch-numberOfWordsInConvolutionalWindowSeen+1	#maxNumberOfWordsInSentenceBatch	#range(0, 1)

	for w in range(wMax):
	
		if(printStatus):
			print("w = " + str(w))
		
		if(resetSequentialInputsIfOnlyFirstInputValid):
			for l in range(1, numberOfLayers+1):
				sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool)

		if(w == 0):
			AfirstLayerShifted = x[:, 0:inputLength]
		else:
			paddings = tf.constant([[0, 0], [w*numberOfFeaturesPerWord, 0]])	#shift input to the right by x words (such that a different input window will be presented to the network)
			#AfirstLayerShifted = x[:, w*numberOfFeaturesPerWord:min(w*numberOfFeaturesPerWord+inputLength, numberOfFeaturesCropped)]
			AfirstLayerShifted = x[:, w*numberOfFeaturesPerWord:w*numberOfFeaturesPerWord+inputLength]
			tf.pad(AfirstLayerShifted, paddings, "CONSTANT")

		
		AfirstLayerShifted = tf.dtypes.cast(AfirstLayerShifted, tf.float32)	#added 01 Sept 2021 #convert input from int to float
			
		AprevLayer = AfirstLayerShifted
		
		if(enforceTcontiguityConstraints):
			TMinPrevLayer, TMaxPrevLayer = TcontiguityInitialiseTemporaryVars(w)
		
		for l in range(1, numberOfLayers+1):	#start algorithm at n_h[1]; ie first hidden layer

			if(printStatus):
				print("\tl = " + str(l))

			#declare variables used across all sequential input of neuron
			#primary vars;
			if(supportSkipLayers):
				if(l == 1):
					AprevLayerAll = AprevLayer	#x
				else:
					AprevLayerAll = tf.concat([AprevLayerAll, AprevLayer], 1)
			if(enforceTcontiguityConstraints):
				TMinPrevLayerAll, TMaxPrevLayerAll = TcontiguityLayerInitialiseTemporaryVars(l, TMinPrevLayer, TMaxPrevLayer)

			#combination vars;
			if(performFunctionOfSequentialInputs):
				#these are all used for different methods of sequential input summation:
				if(performFunctionOfSequentialInputsWeighted):
					if(performFunctionOfSubInputsNonlinear):
						AseqWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)
					else:
						ZseqWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)
				else:
					if(performFunctionOfSubInputsNonlinear):
						AseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)
					else:
						ZseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)

			if(printGraphVisualisation):
				#initialisations required for autograph;
				VseqLast = tf.zeros(Vseq[generateParameterNameSeq(l, 0, "Vseq")].shape, dtype=tf.bool)
				ZseqLast = tf.zeros(Zseq[generateParameterNameSeq(l, 0, "Zseq")].shape)
				AseqLast = tf.zeros(Aseq[generateParameterNameSeq(l, 0, "Aseq")].shape)
				if(enforceTcontiguityConstraints):
					TMaxSeqLast = tf.zeros(TMaxSeq[generateParameterNameSeq(l, 0, "TMaxSeq")].shape, dtype=tf.int32)
					TMinSeqLast = tf.zeros(TMinSeq[generateParameterNameSeq(l, 0, "TMinSeq")].shape, dtype=tf.int32)

			for sForward in range(numberOfSequentialInputs):
				if(useReverseSequentialInputOrder):
					sReverse = numberOfSequentialInputs-sForward-1	
					s = sReverse
				else:
					s = sForward
				
				#for each sequential input of each neuron (stating from last), see if its requirements are satisfied
					#if first sequential input, and hypothetical valid activation of first input, allow reset of neuron sequential input 
						#ie if((s == 0) && (resetSequentialInputsTContiguity)):
							#skip sequential validation requirements as neuron sequential input can be reset
				
				if(printStatus):
					print("\t\ts = " + str(s))

				if(useSparseTensors):
					if(useMultipleSubinputsPerSequentialInput):
						numberSubinputsPerSequentialInput = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)
						#print("numberSubinputsPerSequentialInput = ", numberSubinputsPerSequentialInput)
					
				#identify (hypothetical) activation of neuron sequential input
				if(supportFullConnectivity):
					print("neuralNetworkPropagationSANI error: supportFullConnectivity incomplete")
				else:
					if(useSparseTensors):
						if(supportSkipLayers):
							CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
							CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)					
							AseqInput = tf.gather(AprevLayerAll, CseqCrossLayer, axis=1)
						else:
							AseqInput = tf.gather(AprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
							#print("AprevLayer.shape = ", AprevLayer.shape)
							#print("Cseq.shape = ", Cseq[generateParameterNameSeq(l, s, "Cseq")].shape)
							#print("AseqInput.shape = ", AseqInput.shape)
					else:
						if(supportSkipLayers):
							AseqInput = AprevLayerAll
						else:
							AseqInput = AprevLayer
				if(enforceTcontiguityConstraints):
					TMinSeqInput, TMaxSeqInput = TcontiguitySequentialInputInitialiseTemporaryVars(l, s, TMinPrevLayer, TMaxPrevLayer, TMinPrevLayerAll, TMaxPrevLayerAll)
							
				#calculate validation matrix based upon sequentiality requirements
				#if Vseq[s-1] is True and Vseq[s] is False;
					#OLD: and TMaxSeq[s-1] < w [this condition is guaranteed by processing s in reverse]
				if(s == 0):
					if(resetSequentialInputs):
						if(resetSequentialInputsIfOnlyFirstInputValid):
							VseqExisting = tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")])	#newly activated sequentialActivationFound at higher s, therefore dont reset first s input
						else:
							VseqExisting = tf.fill([batchSize, n_h[l]], True)	#all values of Vseq0_l are always set to 1 as they have no sequential dependencies	
					else:
						if(overwriteSequentialInputs):
							VseqExisting = tf.fill([batchSize, n_h[l]], True)	#all values of Vseq0_l are always set to 1 as they have no sequential dependencies
						else:
							VseqExisting = tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")])		#do not overwrite sequential inputs
				else:
					#if Vseq[s-1] is True and Vseq[s] is False;
						#OLD: and Tseq[s-1] < w [this condition is guaranteed by processing s in reverse]	
					if(useReverseSequentialInputOrder):
						VseqExisting = Vseq[generateParameterNameSeq(l, s-1, "Vseq")]
					else:
						VseqExisting = VseqPrev	#if previous sequentiality check fails, then all future sequentiality checks must fail	
						VseqExisting = tf.math.logical_and(VseqExisting, tf.math.logical_not(VseqPrevNew))	#do not activate current s if previous s was recently activated by same w
					if(not overwriteSequentialInputs):
						VseqExisting = tf.math.logical_and(VseqExisting, tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))	#do not overwrite sequential inputs
						
				VseqFloat = tf.dtypes.cast(VseqExisting, tf.float32)
				
				if(enforceTcontiguityConstraints):
					AseqInput, TMinSeqInputThresholded, TMaxSeqInputThresholded = TcontiguitySequentialInputConstrainAseqInput(l, s, AseqInput, TMinSeqInput, TMaxSeqInput)

				#calculate output for layer sequential input s
				if(allowMultipleSubinputsPerSequentialInput):
					if(useSparseTensors):
						if(performTindependentFunctionOfSubInputs):
							#2. take the AseqInput with the highest weighting
							if(performFunctionOfSubInputsWeighted):
								multiplesSeq = tf.constant([batchSize,1,1], tf.int32)
								WseqTiled = tf.tile(tf.reshape(Wseq[generateParameterNameSeq(l, s, "Wseq")], [1, numberSubinputsPerSequentialInput, n_h[l]]), multiplesSeq)
								AseqInputWeighted = tf.multiply(AseqInput, WseqTiled)
							else:
								AseqInputWeighted = AseqInput

							if(performFunctionOfSubInputsSummation):
								#print("AseqInputWeighted.shape = ", AseqInputWeighted.shape)	# (10, 300, 3000)
								ZseqHypothetical = tf.math.reduce_sum(AseqInputWeighted, axis=1)
							elif(performFunctionOfSubInputsMax):
								#take sub input with max input signal*weight
								ZseqHypothetical = tf.math.reduce_max(AseqInputWeighted, axis=1)
								ZseqHypotheticalIndex = tf.math.argmax(AseqInputWeighted, axis=1)
						else:	
							if(performFunctionOfSubInputsWeighted):
								AseqInputReshaped = AseqInput	#(batchSize, previousLayerFeaturesUsed, layerNeurons)
								AseqInputReshaped = tf.transpose(AseqInputReshaped, [2, 0, 1])	#layerNeurons, batchSize, previousLayerFeaturesUsed
								WseqReshaped = Wseq[generateParameterNameSeq(l, s, "Wseq")]	#previousLayerFeaturesUsed, layerNeurons
								WseqReshaped = tf.transpose(WseqReshaped, [1, 0])	#layerNeurons, previousLayerFeaturesUsed
								WseqReshaped = tf.expand_dims(WseqReshaped, 2)	#layerNeurons, previousLayerFeaturesUsed, 1
								ZseqHypothetical = tf.matmul(AseqInputReshaped, WseqReshaped)	#layerNeurons, batchSize, 1
								ZseqHypotheticalReshaped = ZseqHypothetical
								ZseqHypothetical = tf.squeeze(ZseqHypothetical, axis=2)	#layerNeurons, batchSize
								ZseqHypothetical = tf.transpose(ZseqHypothetical, [1, 0])	#batchSize, layerNeurons
								ZseqHypothetical = tf.add(ZseqHypothetical, Bseq[generateParameterNameSeq(l, s, "Bseq")])	#batchSize, layerNeurons
								#print("ZseqHypothetical.shape = ", ZseqHypothetical.shape)
								#ZseqHypothetical = tf.add(tf.matmul(AseqInput, Wseq[generateParameterNameSeq(l, s, "Wseq")]), Bseq[generateParameterNameSeq(l, s, "Bseq")])
					
								if(debugActivationNormalisation):
									if(w > 0 and l > 1):
										AseqHypothetical, _ = sequentialActivationFunction(ZseqHypothetical)
										#print("WseqReshaped = ", WseqReshaped)
										print("AseqInput.shape = ", AseqInput.shape)
										print("AseqInputReshaped.shape = ", AseqInputReshaped.shape)
										print("WseqReshaped.shape = ", WseqReshaped.shape)
										print("ZseqHypothetical.shape = ", ZseqHypothetical.shape)
										#print("AseqInputReshaped = ", AseqInputReshaped)
										#print("WseqReshaped = ", WseqReshaped)
										#print("ZseqHypotheticalReshaped = ", ZseqHypotheticalReshaped)
										print("AseqInputReshaped[0] = ", AseqInputReshaped[0])
										print("WseqReshaped[0] = ", WseqReshaped[0])
										print("ZseqHypotheticalReshaped[0] = ", ZseqHypotheticalReshaped[0])
										print("tf.math.reduce_mean(AseqInput) = ", tf.math.reduce_mean(AseqInput))
										print("tf.math.reduce_mean(WseqReshaped) = ", tf.math.reduce_mean(WseqReshaped))
										print("tf.math.reduce_mean(ZseqHypothetical) = ", tf.math.reduce_mean(ZseqHypothetical))
										print("tf.math.reduce_mean(AseqHypothetical) = ", tf.math.reduce_mean(AseqHypothetical))
										print("tf.math.zero_fraction(ZseqHypothetical) = ", tf.math.zero_fraction(ZseqHypothetical))
										print("tf.math.zero_fraction(AseqHypothetical) = ", tf.math.zero_fraction(AseqHypothetical))
					
							else:
								print("SANItf2_algorithmSANIsharedModules error: allowMultipleSubinputsPerSequentialInput:useSparseTensors:!performTindependentFunctionOfSubInputs:!performFunctionOfSubInputsWeighted")
								exit(0)
					else:
						if(performSummationOfSubInputsWeighted):
							ZseqHypothetical = tf.add(tf.matmul(AseqInput, Wseq[generateParameterNameSeq(l, s, "Wseq")]), Bseq[generateParameterNameSeq(l, s, "Bseq")])
							#ZseqHypothetical = tf.matmul(AseqInput, Wseq[generateParameterNameSeq(l, s, "Wseq")])							
						elif(performFunctionOfSubInputsAverage):
							AseqInputAverage = tf.math.reduce_mean(AseqInput, axis=1)	#take average
							multiples = tf.constant([1,n_h[l]], tf.int32)
							ZseqHypothetical = tf.tile(tf.reshape(AseqInputAverage, [batchSize, 1]), multiples)
				else:
					ZseqHypothetical = AseqInput	#CHECKTHIS
					
					
				#apply sequential validation matrix
				ZseqHypothetical = tf.multiply(VseqFloat, ZseqHypothetical)
				#print("tf.math.zero_fraction(ZseqHypothetical) after sequential validation matrix = ", tf.math.zero_fraction(ZseqHypothetical))
							
				#threshold output/check output threshold
				ZseqThresholded, ZseqPassThresold = sequentialActivationFunction(ZseqHypothetical)			
				
				if(enforceTcontiguityConstraints):
					TcontiguityUpdateArrays(l, s, ZseqPassThresold, TMinSeqInputThresholded, TMaxSeqInputThresholded)
				
				ZseqPassThresoldInt = tf.dtypes.cast(ZseqPassThresold, tf.int32)
				ZseqPassThresoldNot = tf.math.logical_not(ZseqPassThresold)
				ZseqPassThresoldNotInt = tf.dtypes.cast(ZseqPassThresoldNot, tf.int32)
					
				#reset appropriate neurons
				if((resetSequentialInputs) and (s == 0)):
					resetRequiredMatrix = tf.math.logical_and(ZseqPassThresold, Vseq[generateParameterNameSeq(l, s, "Vseq")])	#reset sequential inputs if first input valid and first input has already been activated
					if(resetSequentialInputsIfOnlyFirstInputValid):
						resetRequiredMatrix = tf.math.logical_and(resetRequiredMatrix, tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")]))	#do not reset sequential inputs if a higher sequential input was newly activated
					for s2 in range(numberOfSequentialInputs):
						Vseq[generateParameterNameSeq(l, s2, "Vseq")] = tf.dtypes.cast(tf.multiply(tf.dtypes.cast(Vseq[generateParameterNameSeq(l, s2, "Vseq")], tf.int32), tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32)), tf.bool)
						if(recordNetworkWeights):
							if(recordSubInputsWeighted):
								numberSubinputsPerSequentialInput2 = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s2)
								multiples = tf.constant([1,numberSubinputsPerSequentialInput2,1], tf.int32)
								resetRequiredMatrixTiled = tf.tile(tf.reshape(resetRequiredMatrix, [batchSize, 1, n_h[l]]), multiples)
								AseqInputVerified[generateParameterNameSeq(l, s2, "AseqInputVerified")] = tf.math.logical_and(AseqInputVerified[generateParameterNameSeq(l, s2, "AseqInputVerified")], tf.math.logical_not(resetRequiredMatrixTiled))						
						if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
							if(enforceTcontiguityConstraints):
								TMaxSeq[generateParameterNameSeq(l, s2, "TMaxSeq")] = tf.multiply(TMaxSeq[generateParameterNameSeq(l, s2, "TMaxSeq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
								TMinSeq[generateParameterNameSeq(l, s2, "TMinSeq")] = tf.multiply(TMinSeq[generateParameterNameSeq(l, s2, "TMinSeq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							Zseq[generateParameterNameSeq(l, s2, "Zseq")] = tf.multiply(Zseq[generateParameterNameSeq(l, s2, "Zseq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
							Aseq[generateParameterNameSeq(l, s2, "Aseq")] = tf.multiply(Aseq[generateParameterNameSeq(l, s2, "Aseq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
					if(resetSequentialInputsIfOnlyFirstInputValid):
						sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_and(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], tf.math.logical_not(resetRequiredMatrix))
					if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
						Z[generateParameterName(l, "Z")] = tf.multiply(Z[generateParameterName(l, "Z")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
						A[generateParameterName(l, "A")] = tf.multiply(A[generateParameterName(l, "A")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
						if(enforceTcontiguityConstraints):
							TMax[generateParameterName(l, "TMax")] = tf.multiply(TMax[generateParameterName(l, "TMax")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							TMin[generateParameterName(l, "TMin")] = tf.multiply(TMin[generateParameterName(l, "TMin")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))


				#calculate updated Vseq/Zseq/Aseq activation matrix taking into account previously activated sectors (batchIndices, neurons):
				VseqExistingOld = tf.multiply(tf.dtypes.cast(Vseq[generateParameterNameSeq(l, s, "Vseq")], tf.float32), tf.dtypes.cast(ZseqPassThresoldNotInt, tf.float32))	#zero all Vseq sectors (batchIndices, neurons) which pass threshold; prepare for addition
				VseqUpdated = SANItf2_algorithmSANIoperations.updateTensorCells(VseqExistingOld, tf.cast(ZseqPassThresold, tf.float32))	#VseqUpdated = tf.math.logical_or(Vseq[generateParameterNameSeq(l, s, "Vseq")], ZseqPassThresold)
				VseqUpdated = tf.cast(VseqUpdated, tf.bool)
				ZseqExistingOld = tf.multiply(Zseq[generateParameterNameSeq(l, s, "Zseq")], tf.dtypes.cast(ZseqPassThresoldNotInt, tf.float32))	#zero all Zseq sectors (batchIndices, neurons) which pass threshold; prepare for addition
				ZseqUpdated = SANItf2_algorithmSANIoperations.updateTensorCells(ZseqExistingOld, ZseqThresholded)	#tf.add(ZseqExistingOld, ZseqThresholded)
				AseqUpdated, _ = sequentialActivationFunction(ZseqUpdated)	
														
				Aseq[generateParameterNameSeq(l, s, "Aseq")] = AseqUpdated
				Zseq[generateParameterNameSeq(l, s, "Zseq")] = ZseqUpdated
				Vseq[generateParameterNameSeq(l, s, "Vseq")] = VseqUpdated
				
				if(resetSequentialInputsIfOnlyFirstInputValid):
					if(s > 0):
						sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_or(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], ZseqPassThresold)	#record that a higher sequential input was newly activated
						
															
				#apply weights to input of neuron sequential input
				if(performFunctionOfSequentialInputs):
					#these are all used for different methods of sequential input summation
					if(performFunctionOfSequentialInputsWeighted):	#note performFunctionOfSequentialInputsWeighted is mathematically equivalent to performFunctionOfSequentialInputsSummationWeighted
						multiples = tf.constant([batchSize,1], tf.int32)
						Wtiled = tf.tile(tf.reshape(W[generateParameterName(l, "W")][s], [1, n_h[l]]), multiples)
						if(performFunctionOfSubInputsNonlinear):
							AseqWeighted = tf.multiply(AseqUpdated, Wtiled)
							AseqWeightedSum = tf.math.add(AseqWeightedSum, AseqWeighted)	
						else:
							ZseqWeighted = tf.multiply(ZseqUpdated, Wtiled)
							ZseqWeightedSum = tf.math.add(ZseqWeightedSum, ZseqWeighted)
					else:
						if(performFunctionOfSubInputsNonlinear):
							AseqSum = tf.add(AseqSum, AseqUpdated)
						else:
							ZseqSum = tf.add(ZseqSum, ZseqUpdated)			

				if(s == numberOfSequentialInputs-1):
					ZseqLast = ZseqUpdated
					AseqLast = AseqUpdated
					VseqLast = VseqUpdated
					if(enforceTcontiguityConstraints):
						TMaxSeqLast = TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")]
						TMinSeqLast = TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")]

				VseqPrev = VseqUpdated
				VseqPrevNew = ZseqPassThresold
			
				recordActivitySequentialInput(l, s, ZseqPassThresold)
				
			if(performFunctionOfSequentialInputs):
				if(performFunctionOfSequentialInputsWeighted):
					if(performFunctionOfSubInputsNonlinear):
						Z1 = AseqWeightedSum			
					else:
						Z1 = ZseqWeightedSum			
				else:
					if(performFunctionOfSubInputsNonlinear):
						Z1 = AseqSum			
					else:
						Z1 = ZseqSum
				if(sequentialInputCombinationModeSummationAveraged):
					Z1 = Z1/numberOfSequentialInputs
				if(performFunctionOfSequentialInputsNonlinear):
					A1 = activationFunction(Z1)
				else:
					A1 = Z1	
				if(performFunctionOfSequentialInputsVerify):
					Z1 = tf.multiply(Z1, tf.dtypes.cast(VseqLast, tf.float32))
					A1 = tf.multiply(A1, tf.dtypes.cast(VseqLast, tf.float32))
					
				#!performFunctionOfSequentialInputsSummation & performFunctionOfSequentialInputsWeighted; mathemetically equivalent alternate calculation method:
				#Aseq: batchSize, layerNeurons;	for each s
				#AseqReshaped: layerNeurons, batchSize, numberOfSequentialInputs
				#W: numberOfSequentialInputs,layerNeurons
				#Wreshaped: layerNeurons, numberOfSequentialInputs, 1
				#Z = tf.matmul(AseqReshaped, Wreshaped)	#layerNeurons, batchSize, 1
				#Z = tf.squeeze(Z, axis=2)	#layerNeurons, batchSize
				#Z = tf.transpose(Z, [1, 0])	#batchSize, layerNeurons
				#Z = tf.add(Z, B[generateParameterName(l, "B")])	#batchSize, layerNeurons
				#A = activationFunction(Z)	#batchSize, layerNeurons
			else:
				#VseqLastFloat = VseqFloat
				Z1 = ZseqLast
				A1 = AseqLast
			
			if(debugActivationNormalisation):
				if(l > 1):
					#print("A1 = ", A1)
					print("tf.math.reduce_mean(A1) = ", tf.math.reduce_mean(A1))
					print("tf.math.zero_fraction(A1) = ", tf.math.zero_fraction(A1))
				print("tf.math.zero_fraction(A1) = ", tf.math.zero_fraction(A1))
			
			A[generateParameterName(l, "A")] = A1
			Z[generateParameterName(l, "Z")] = Z1
			
			if(enforceTcontiguityConstraints):
				TMax[generateParameterName(l, "TMax")] = TMaxSeqLast
				TMin[generateParameterName(l, "TMin")] = TMinSeqLast
				TMaxPrevLayer = TMax[generateParameterName(l, "TMax")]
				TMinPrevLayer = TMin[generateParameterName(l, "TMin")]			
			
			AprevLayer = A[generateParameterName(l, "A")]

		
	ZlastLayer = Z[generateParameterName(numberOfLayers, "Z")]
	
	if(debugActivationNormalisation):
		#print("ZlastLayer = ", ZlastLayer)
		print("tf.math.reduce_mean(ZlastLayer) = ", tf.math.reduce_mean(ZlastLayer))
		print("tf.math.zero_fraction(ZlastLayer) = ", tf.math.zero_fraction(ZlastLayer))
	
	#print("ZlastLayer = ", ZlastLayer)
	
	if(useLearningRuleBackpropagation):
		pred = SANItf2_algorithmSANIoperations.generatePrediction(ZlastLayer, Whead, applySoftmax=(not vectorisedOutput))
		if(printIO):
			print("pred = ", pred)
	else:
		if(enforceTcontiguityConstraints):
			ZlastLayer = TcontiguityCalculateOutput(ZlastLayer)
		pred = ZlastLayer
			
	return pred
				
def sequentialActivationFunction(Zseq):
	#threshold output/check output threshold
	if(performThresholdOfSubInputsNonlinear):
		ZseqThresholded = activationFunction(Zseq)	#Aseq
		ZseqPassThresold = tf.math.greater(ZseqThresholded, 0.0)
	else:
		ZseqPassThresold = tf.math.greater(Zseq, sequentialInputActivationThreshold)
		ZseqThresholded = tf.multiply(Zseq, tf.dtypes.cast(ZseqPassThresold, tf.float32))	#Aseq	

	return ZseqThresholded, ZseqPassThresold	
	
def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A				




def TcontiguityInitialiseTemporaryVars(w):
	TMinPrevLayer, TMaxPrevLayer = (None, None)
	
	TMaxPrevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
	TMaxPrevLayer = TMaxPrevLayer*w
	TMinPrevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
	TMinPrevLayer = TMinPrevLayer*w

	return TMinPrevLayer, TMaxPrevLayer
	
def TcontiguityLayerInitialiseTemporaryVars(l, TMinPrevLayer, TMaxPrevLayer):
	TMinPrevLayerAll, TMaxPrevLayerAll = (None, None)
	
	#declare variables used across all sequential input of neuron
	#primary vars;
	if(l == 1):
		if(supportSkipLayers):
			TMaxPrevLayerAll = TMaxPrevLayer
			TMinPrevLayerAll = TMinPrevLayer
	else:
		if(supportSkipLayers):
			TMaxPrevLayerAll = tf.concat([TMaxPrevLayerAll, TMaxPrevLayer], 1)
			TMinPrevLayerAll = tf.concat([TMinPrevLayerAll, TMinPrevLayer], 1)

	return TMinPrevLayerAll, TMaxPrevLayerAll


#declare variables used across all sequential input of neuron
def TcontiguitySequentialInputInitialiseTemporaryVars(l, s, TMinPrevLayer, TMaxPrevLayer, TMinPrevLayerAll, TMaxPrevLayerAll):
	TMinSeqInput, TMaxSeqInput = (None, None)
	
	#identify (hypothetical) activation of neuron sequential input
	if(supportFullConnectivity):
		print("neuralNetworkPropagationSANI error: supportFullConnectivity incomplete")
	else:
		if(useSparseTensors):
			if(supportSkipLayers):
				CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
				CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)						
				if(allowMultipleContributingSubinputsPerSequentialInput):
					if((resetSequentialInputsTContiguity) and (s == 0)):
						AseqInputTadjusted = tf.gather(tf.multiply(AprevLayerAll, tf.dtypes.cast(TprevLayerAll, tf.float32)), CseqCrossLayer, axis=1) 
				else:
					TMaxSeqInput = tf.gather(TMaxPrevLayerAll, CseqCrossLayer, axis=1)
					TMinSeqInput = tf.gather(TMinPrevLayerAll, CseqCrossLayer, axis=1)
			else:
				if(allowMultipleContributingSubinputsPerSequentialInput):
					if((resetSequentialInputsTContiguity) and (s == 0)):
						AseqInputTadjusted = tf.gather(tf.multiply(AprevLayer, tf.dtypes.cast(TprevLayer, tf.float32)), Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
				else:
					TMaxSeqInput = tf.gather(TMaxPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
					TMinSeqInput = tf.gather(TMinPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
		else:
			if(supportSkipLayers):
				if(allowMultipleContributingSubinputsPerSequentialInput):
					if((resetSequentialInputsTContiguity) and (s == 0)):
						AseqInputTadjusted = tf.multiply(AprevLayerAll, tf.dtypes.cast(TprevLayerAll, tf.float32))
				else:
					TMaxSeqInput = TMaxPrevLayerAll
					TMinSeqInput = TMinPrevLayerAll
			else:
				if(allowMultipleContributingSubinputsPerSequentialInput):
					if((resetSequentialInputsTContiguity) and (s == 0)):
						AseqInputTadjusted = tf.multiply(AprevLayer, tf.dtypes.cast(TprevLayer, tf.float32))
				else:
					TMaxSeqInput = TMaxPrevLayer
					TMinSeqInput = TMinPrevLayer

	return TMinSeqInput, TMaxSeqInput


def TcontiguitySequentialInputConstrainAseqInput(l, s, AseqInput, TMinSeqInput, TMaxSeqInput):
	AseqInput, TMinSeqInputThresholded, TMaxSeqInputThresholded = (None, None, None)

	if(allowMultipleSubinputsPerSequentialInput):
		if(useSparseTensors):
			if(not allowMultipleContributingSubinputsPerSequentialInput):
				#ensure that T continguous constraint is met (T threshold AseqInput);
				#NO: take the max subinput pathway only (ie matrix mult but with max() rather than sum() for each dot product)
				if(s > 0):
					numberSubinputsPerSequentialInput = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)
					multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

					if(enforceTcontiguityBetweenSequentialInputs):
						#1. first select the AseqInput inputs that have TMaxSeq[l, s] contiguous with TMaxSeq[l, s-1]:
						TMaxSeqPrevPlus1 = TMaxSeq[generateParameterNameSeq(l, s-1, "TMaxSeq")]+1

						TMaxSeqPrevPlus1Tiled = tf.tile(tf.reshape(TMaxSeqPrevPlus1, [batchSize, 1, n_h[l]]), multiples)
						TMinSeqInputThreshold = tf.math.equal(TMinSeqInput, TMaxSeqPrevPlus1Tiled)
						AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMinSeqInputThreshold, tf.float32))

						TMinSeqInputThresholded = TMinSeqPrev	#note this is recording the min of the sequence, not the sequential input	#OLD: TMaxSeqPrevPlus1

						AseqInput = AseqInputTthresholded

					if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
						TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
						AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMaxSeqInputThreshold, tf.float32))

						TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w

						AseqInput = AseqInputTthresholded

						#VseqFloatTiled = tf.tile(tf.reshape(VseqFloat, [batchSize, 1, n_h[l]]), multiples)
						#AseqInputThresholded = tf.multiply(VseqFloatTiled, AseqInput)	#done later
				else:
					if(resetSequentialInputsTContiguity):
						#NO: ZseqHypotheticalResetThreshold = 1 	#always reset if valid first input (CHECKTHIS)	#dummy variable
						#only reset first sequential input if TMaxSeqInput > TMax[l]

						numberSubinputsPerSequentialInput = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)
						multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

						TMaxtiled = tf.tile(tf.reshape(TMax[generateParameterName(l, "TMax")], [batchSize, 1, n_h[l]]), multiples)
						VseqTiled = tf.tile(tf.reshape(Vseq[generateParameterNameSeq(l, s, "Vseq")], [batchSize, 1, n_h[l]]), multiples)

						ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(VseqTiled), tf.math.greater(TMaxSeqInput, tf.dtypes.cast(TMaxtiled, tf.int32)))
						AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(ZseqHypotheticalResetThreshold, tf.float32))

						AseqInput = AseqInputTthresholded

						if(enforceTcontiguityTakeEncapsulatedSubinputWithMinTvalue):
							TMinSeqInputThresholdIndices = tf.dtypes.cast(tf.math.argmin(TMinSeqInput, axis=1), tf.int32)

							#AseqInput shape: batchSize*numSubinputs*numNeuronsOnLayer
							AseqInputReordedAxes = tf.transpose(AseqInput, [0, 2, 1])
							AseqInputReordedAxesFlattened = tf.reshape(AseqInputReordedAxes, [AseqInputReordedAxes.shape[0]*AseqInputReordedAxes.shape[1], AseqInputReordedAxes.shape[2]])
							idx_0 = tf.reshape(tf.range(AseqInputReordedAxesFlattened.shape[0]), TMinSeqInputThresholdIndices.shape)
							indices=tf.stack([idx_0,TMinSeqInputThresholdIndices],axis=-1)
							AseqInputTthresholded = tf.gather_nd(AseqInputReordedAxesFlattened, indices)
							AseqInputTthresholded = tf.expand_dims(AseqInputTthresholded, axis=1)	#note this dimension will be reduced/eliminated below so its size does not matter

							TMinSeqInputThresholded = tf.math.reduce_min(TMinSeqInput, axis=1)

							AseqInput = AseqInputTthresholded

						if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
							TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
							AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMaxSeqInputThreshold, tf.float32))

							TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w

							AseqInput = AseqInputTthresholded
			else:
				print("TcontiguitySequentialInputConstrainAseqInput error: TcontiguitySequentialInputConstrainAseqInput:allowMultipleSubinputsPerSequentialInput & !useSparseTensors")
				exit()
	else:
		#ensure that T continguous constraint is met (T threshold AseqInput);
		if(s > 0):
			if(enforceTcontiguityBetweenSequentialInputs):
				#1. first select the AseqInput inputs that have TMaxSeq[l, s] contiguous with TMaxSeq[l, s-1]:

				TMaxSeqPrevPlus1 = TMaxSeq[generateParameterNameSeq(l, s-1, "TMaxSeq")]+1
				TseqInputThreshold = tf.math.equal(TMinSeqInput, TMaxSeqPrevPlus1)

				#print("AseqInput = ", AseqInput)
				AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TseqInputThreshold, tf.float32))

				TMinSeqInputThresholded = TMaxSeqPrevPlus1

				AseqInput = AseqInputTthresholded

			if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
				TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
				AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMaxSeqInputThreshold, tf.float32))

				TMaxSeqInputThresholded = TMaxSeqInput	#CHECKTHIS - ensure equal to w

				AseqInput = AseqInputTthresholded
		else:
			if(resetSequentialInputsTContiguity):
				#only reset first sequential input if TMaxSeqInput > T[l]

				ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]), tf.math.greater(TMaxSeqInput, tf.dtypes.cast(TMax[generateParameterName(l, "TMax")], tf.int32)))

				AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(ZseqHypotheticalResetThreshold, tf.float32))
				AseqInput = AseqInputTthresholded

				if(enforceTcontiguityTakeEncapsulatedSubinputWithMinTvalue):
					TMinSeqInputThresholded = TMinSeqInput

				if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
					TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
					AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMaxSeqInputThreshold, tf.float32))

					TMaxSeqInputThresholded = TMaxSeqInput	#CHECKTHIS - ensure equal to w

					AseqInput = AseqInputTthresholded
					
	return AseqInput, TMinSeqInputThresholded, TMaxSeqInputThresholded

def TcontiguityUpdateArrays(l, s, ZseqPassThresold, TMinSeqInputThresholded, TMaxSeqInputThresholded):
	#update ZseqPassThresold based on reset
	if(allowMultipleSubinputsPerSequentialInput):
		if(allowMultipleContributingSubinputsPerSequentialInput):
			if((resetSequentialInputsTContiguity) and (s == 0)):
				#ensure that T continguous constraint is met (T threshold ZseqPassThresold);

				#calculate output for layer sequential input s
				#old slow: only factor in inputs that have changed since the component was originally activated; if Torigin > TMaxSeq[s]
				#new: only factor in inputs if ZseqHypotheticalTadjusted > ZseqTadjusted+1
				ZseqHypotheticalTadjusted = tf.add(tf.matmul(AseqInputTadjusted, Wseq[generateParameterNameSeq(l, s, "Wseq")]), Bseq[generateParameterNameSeq(l, s, "Bseq")])
				ZseqHypotheticalTadjusted = tf.divide(ZseqHypotheticalTadjusted, AseqInputTadjusted.shape[1])	#normalise T adjustment
				ZseqHypotheticalTadjustedResetThreshold = tf.math.logical_or(tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]), tf.math.greater(ZseqHypotheticalTadjusted, ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")]+averageTimeChangeOfNewInputRequiredForReset))

				ZseqPassThresold = tf.math.logical_and(ZseqPassThresold, ZseqHypotheticalTadjustedResetThreshold)	#ensures that if reset is required, Tadjusted threshold is met

				#update thresholded output;
				ZseqThresholded = tf.multiply(ZseqThresholded, tf.dtypes.cast(ZseqPassThresold, tf.float32))	

	ZseqPassThresoldInt = tf.dtypes.cast(ZseqPassThresold, tf.int32)
	ZseqPassThresoldNot = tf.math.logical_not(ZseqPassThresold)
	ZseqPassThresoldNotInt = tf.dtypes.cast(ZseqPassThresoldNot, tf.int32)

	TMaxSeqPassThresoldUpdatedValues = tf.multiply(ZseqPassThresoldInt, TMaxSeqInputThresholded)
	TMaxSeqExistingOld = tf.multiply(TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")], ZseqPassThresoldNotInt)
	TMaxSeqUpdated = SANItf2_algorithmSANIoperations.updateTensorCells(TMaxSeqExistingOld, TMaxSeqPassThresoldUpdatedValues)	#tf.add(TMaxSeqExistingOld, TMaxSeqPassThresoldUpdatedValues)

	TMinSeqPassThresoldUpdatedValues = tf.multiply(ZseqPassThresoldInt, TMinSeqInputThresholded)
	TMinSeqExistingOld = tf.multiply(TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")], ZseqPassThresoldNotInt)
	TMinSeqUpdated = SANItf2_algorithmSANIoperations.updateTensorCells(TMinSeqExistingOld, TMinSeqPassThresoldUpdatedValues)	#tf.add(TMinSeqExistingOld, TMinSeqPassThresoldUpdatedValues)

	if(allowMultipleContributingSubinputsPerSequentialInput):
		if((resetSequentialInputsTContiguity) and (s == 0)):
			ZseqTadjustedThresholded = tf.multiply(ZseqHypotheticalTadjusted, tf.dtypes.cast(ZseqPassThresoldInt, tf.float32))
			ZseqTadjustedUpdatedOld = tf.multiply(ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")], tf.dtypes.cast(ZseqPassThresoldNotInt, tf.float32))
			ZseqTadjustedUpdated = SANItf2_algorithmSANIoperations.updateTensorCells(ZseqTadjustedUpdatedOld, ZseqTadjustedThresholded)	#ZseqTadjustedUpdated = tf.add(ZseqTadjustedUpdated, ZseqTadjustedThresholded)
			ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")] = ZseqTadjustedUpdated

	#update parameter storage;
	TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = TMaxSeqUpdated
	TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = TMinSeqUpdated

def TcontiguityCalculateOutput(ZlastLayer):
	if(enforceTcontiguityStartAndEndOfSequence):
		TMaxLastLayer = TMax[generateParameterName(numberOfLayers, "TMax")] 
		TMinLastLayer = TMin[generateParameterName(numberOfLayers, "TMin")]
		TMaxLastLayerThreshold = tf.math.equal(TMaxLastLayer, w-1)
		TMinLastLayerThreshold = tf.math.equal(TMinLastLayer, 0)
		ZlastLayer = tf.multiply(ZlastLayer, tf.dtypes.cast(TMaxLastLayerThreshold, tf.float32))
		ZlastLayer = tf.multiply(ZlastLayer, tf.dtypes.cast(TMinLastLayerThreshold, tf.float32))
	return ZlastLayer
	
def recordActivitySequentialInput(l, s, ZseqPassThresold):
	if(recordNetworkWeights):
		if(s == numberOfSequentialInputs-1):
			if(recordSubInputsWeighted):
				for s2 in range(numberOfSequentialInputs):
					numberSubinputsPerSequentialInput2 = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s2)
					multiples = tf.constant([1,numberSubinputsPerSequentialInput2,1], tf.int32)
					neuronNewlyActivatedTiled = tf.tile(tf.reshape(ZseqPassThresold, [batchSize, 1, n_h[l]]), multiples)	#or; VseqBool (since will be logically anded with AseqInputVerified)

					AseqInputVerifiedAndNeuronActivated = tf.math.logical_and(AseqInputVerified[generateParameterNameSeq(l, s2, "AseqInputVerified")], neuronNewlyActivatedTiled)
					AseqInputVerifiedAndNeuronActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(AseqInputVerifiedAndNeuronActivated, tf.float32), axis=0)

					if(recordSequentialInputsWeighted):
						WRseq[generateParameterNameSeq(l, s2, "WRseq")] = tf.add(WRseq[generateParameterNameSeq(l, s2, "WRseq")], AseqInputVerifiedAndNeuronActivatedBatchSummed)

				if(recordNeuronsWeighted):
					#record neuron fires if last sequential input fires
					neuronNewlyActivated = ZseqPassThresold	
					neuronNewlyActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(neuronNewlyActivated, tf.float32), axis=0)
					BR[generateParameterName(l, "BR")] = tf.add(BR[generateParameterName(l, "BR")], neuronNewlyActivatedBatchSummed)
					
		if(recordSequentialInputsWeighted):	
			#record current sequential input fires
			sequentialInputNewlyActivated = ZseqPassThresold
			sequentialInputNewlyActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(sequentialInputNewlyActivated, tf.float32), axis=0)
			WR[generateParameterName(l, "WR")] = tf.add(WR[generateParameterName(l, "WR")][s], sequentialInputNewlyActivatedBatchSummed)


	
	
