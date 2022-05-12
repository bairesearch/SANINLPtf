"""SANItf2_algorithmSANIsharedModulesBinary.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SANItf2.py

# Usage:
see SANItf2.py

# Description:
SANItf algorithm SANI shared modules binary - define Sequentially Activated Neuronal Input neural network with shared modules and binary weights and activation signals

See shared modules

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
if(algorithmSANI == "sharedModulesBinary"):
	Vseq = {}
	Zseq = {}
	Aseq = {}
	Z = {}
	A = {}
	if(enforceTcontiguityConstraints):
		TMaxSeq = {}
		TMinSeq = {}
		ZseqTadjusted = {}
		T = {}
		TMax = {}
		TMin = {}
	if(resetSequentialInputsIfOnlyFirstInputValid):
		sequentialActivationFound = {}	#records whether last s sequential input was activated
	AseqInputVerified = {}

#end common SANItf2_algorithmSANI.py code


def neuralNetworkPropagation(x, networkIndex=None):
	return neuralNetworkPropagationSANI(x)				
							
def neuralNetworkPropagationSANI(x):
		
	#note connectivity indexes are used rather than sparse weight matrices due to limitations in current tf2 sparse tensor implementation
	
	#definitions for reference:
	
	#neuron sequential input vars;
	#x/AprevLayer	#output vector (dim: batchSize*n_h[l])
	#Cseq	#static connectivity matrix (int) - indexes of neurons on prior layer stored; mapped to W  (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
	#if(supportSkipLayers):
		#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
	#AseqSum	#combination variable
	#Vseq	#mutable verification vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)			#records whether particular neuron sequential inputs are currently active
	#Zseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#Aseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs

	#neuron vars;
	#Q  
	#Z	#neuron activation function input (dim: batchSize*n_h[l])
	#A	#neuron activation function output (dim: batchSize*n_h[l])
	
	#Tcontiguity vars;
	#if(enforceTcontiguityConstraints):
		#neuron sequential input vars;
		#TMaxSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a particular sequential input last fired
		#TMinSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a first encapsulated subinput fired
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
				
	x = tf.dtypes.cast(x, tf.int32)	#SANItf2_algorithmSANIsharedModulesBinary expects input as int (loadDatasetTypeX returns float by default)
	
	batchSize = x.shape[0]
	
	#optimise feed length based on max sentence length in batch:
	#unoptimised: numberOfFeatures = x.shape[1]
	
	xIsNotPadding = tf.math.not_equal(x, tf.dtypes.cast(paddingTagIndex, tf.int32)) 
	coordinatesOfNotPadding = tf.where(xIsNotPadding)		
	numberOfFeaturesCropped = tf.reduce_max(coordinatesOfNotPadding[:, 1])
	numberOfFeaturesCropped = tf.dtypes.cast(numberOfFeaturesCropped, tf.int32)
	numberOfFeaturesCropped = tf.add(numberOfFeaturesCropped, 1)
	maxNumberOfWordsInSentenceBatch = tf.divide(numberOfFeaturesCropped, numberOfFeaturesPerWord)
	maxNumberOfWordsInSentenceBatch = tf.dtypes.cast(maxNumberOfWordsInSentenceBatch, tf.int32)
		
	inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen
		
	match_indices = tf.where(tf.equal(paddingTagIndex, x), x=tf.range(tf.shape(x)[1])*tf.ones_like(x), y=(tf.shape(x)[1])*tf.ones_like(x))
	numberOfFeaturesActiveBatch = tf.math.argmin(match_indices, axis=1)
	numberOfWordsInSentenceBatch = tf.divide(numberOfFeaturesActiveBatch, numberOfFeaturesPerWord)
	numberOfWordsInSentenceBatch = tf.dtypes.cast(numberOfWordsInSentenceBatch, tf.int32)
	
	for l in range(1, numberOfLayers+1):
		Z[generateParameterName(l, "Z")] = tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool)	#A=Z for binary activations
		A[generateParameterName(l, "A")] = tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool)	#A=Z for binary activations
		if(enforceTcontiguityConstraints):
			TMax[generateParameterName(l, "TMax")] = tf.zeros([batchSize, n_h[l]], dtype=tf.int32)
			TMin[generateParameterName(l, "TMin")] = tf.zeros([batchSize, n_h[l]], dtype=tf.int32)
		for s in range(numberOfSequentialInputs):
			Vseq[generateParameterNameSeq(l, s, "Vseq")] = tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool)
			Zseq[generateParameterNameSeq(l, s, "Zseq")] = tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool)	#A=Z for binary activations
			Aseq[generateParameterNameSeq(l, s, "Aseq")] = tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool)	#A=Z for binary activations
			if(enforceTcontiguityConstraints):
				TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = tf.zeros([batchSize, n_h[l]], dtype=tf.int32)
				TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = tf.zeros([batchSize, n_h[l]], dtype=tf.int32)
			if(recordNetworkWeights):
				if(recordSubInputsWeighted):
					numberSubinputsPerSequentialInput = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)
					AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")] = tf.dtypes.cast(tf.zeros([batchSize, numberSubinputsPerSequentialInput, n_h[l]]), dtype=tf.bool)
		
	wMax = maxNumberOfWordsInSentenceBatch-numberOfWordsInConvolutionalWindowSeen+1
				
	for w in range(wMax):
	
		if(printStatus):
			print("w = " + str(w))
		
		if(resetSequentialInputsIfOnlyFirstInputValid):
			for l in range(1, numberOfLayers+1):
				sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool)

		if(w == 0):
			AfirstLayerShifted =  tf.dtypes.cast(x[:, 0:inputLength], tf.bool)
		else:
			paddings = tf.constant([[0, 0], [w*numberOfFeaturesPerWord, 0]])	#shift input to the right by x words (such that a different input window will be presented to the network)
			AfirstLayerShifted = tf.dtypes.cast(x[:, w*numberOfFeaturesPerWord:w*numberOfFeaturesPerWord+inputLength], tf.bool)
			tf.pad(AfirstLayerShifted, paddings, "CONSTANT")
		
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
						
				#identify (hypothetical) activation of neuron sequential input
				if(supportFullConnectivity):
					print("neuralNetworkPropagationSANI error: supportFullConnectivity incomplete")
				else:
					if(supportSkipLayers):
						CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
						CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)
						AseqInput = tf.gather(AprevLayerAll, CseqCrossLayer, axis=1)
					else:
						AseqInput = tf.gather(AprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
				if(enforceTcontiguityConstraints):
					TMinSeqInput, TMaxSeqInput = TcontiguitySequentialInputInitialiseTemporaryVars(l, s, TMinPrevLayer, TMaxPrevLayer, TMinPrevLayerAll, TMaxPrevLayerAll)
				
				#calculate validation matrix based upon sequentiality requirements			
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
		

				VseqBool = VseqExisting
									
				#calculate output for layer sequential input s

				if(enforceTcontiguityConstraints):
					AseqInput, TMinSeqInputThresholded, TMaxSeqInputThresholded = TcontiguitySequentialInputConstrainAseqInput(l, s, AseqInput, TMinSeqInput, TMaxSeqInput)

				#take any active and Tthreshold valid sub input:
				#CHECKTHIS - for performTindependentFunctionOfSubInputs?; consider adding performFunctionOfSubInputsWeighted:Wseq/Bseq option
				if(allowMultipleSubinputsPerSequentialInput):
					ZseqHypothetical = tf.math.reduce_any(AseqInput, axis=1)				
				else:
					ZseqHypothetical = AseqInput
				
				#apply sequential validation matrix
				ZseqHypothetical = tf.math.logical_and(VseqBool, ZseqHypothetical)
								
				#threshold output/check output threshold
				_, ZseqPassThresold = sequentialActivationFunction(ZseqHypothetical)
				
				if(enforceTcontiguityConstraints):
					TcontiguityUpdateArrays(l, s, ZseqPassThresold, TMaxSeqInputThresholded, TMinSeqInputThresholded)

				ZseqPassThresoldInt = tf.dtypes.cast(ZseqPassThresold, tf.int32)
				ZseqPassThresoldNot = tf.math.logical_not(ZseqPassThresold)
				ZseqPassThresoldNotInt = tf.dtypes.cast(ZseqPassThresoldNot, tf.int32)
															
				#reset appropriate neurons
				if((resetSequentialInputs) and (s == 0)):
					resetRequiredMatrix = tf.math.logical_and(ZseqPassThresold, Vseq[generateParameterNameSeq(l, s, "Vseq")])	#reset sequential inputs if first input valid and first input has already been activated
					if(resetSequentialInputsIfOnlyFirstInputValid):
						resetRequiredMatrix = tf.math.logical_and(resetRequiredMatrix, tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")]))	#do not reset sequential inputs if a higher sequential input was newly activated
					for s2 in range(numberOfSequentialInputs):
						Vseq[generateParameterNameSeq(l, s2, "Vseq")] = tf.math.logical_and(Vseq[generateParameterNameSeq(l, s2, "Vseq")], tf.math.logical_not(resetRequiredMatrix))
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
							Zseq[generateParameterNameSeq(l, s2, "Zseq")] = tf.math.logical_and(Zseq[generateParameterNameSeq(l, s2, "Zseq")], tf.math.logical_not(resetRequiredMatrix))
							Aseq[generateParameterNameSeq(l, s2, "Aseq")] = tf.math.logical_and(Aseq[generateParameterNameSeq(l, s2, "Aseq")], tf.math.logical_not(resetRequiredMatrix))
					if(resetSequentialInputsIfOnlyFirstInputValid):
						sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_and(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], tf.math.logical_not(resetRequiredMatrix))
					if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
						Z[generateParameterName(l, "Z")] = tf.math.logical_and(Z[generateParameterName(l, "Z")], tf.math.logical_not(resetRequiredMatrix))
						A[generateParameterName(l, "A")] = tf.math.logical_and(A[generateParameterName(l, "A")], tf.math.logical_not(resetRequiredMatrix))
						if(enforceTcontiguityConstraints):
							TMax[generateParameterName(l, "TMax")] = tf.multiply(TMax[generateParameterName(l, "TMax")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							TMin[generateParameterName(l, "TMin")] = tf.multiply(TMin[generateParameterName(l, "TMin")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))

				#calculate updated Vseq/Zseq/Aseq activation matrix taking into account previously activated sectors (batchIndices, neurons):
				VseqExistingOld = tf.multiply(tf.dtypes.cast(Vseq[generateParameterNameSeq(l, s, "Vseq")], tf.float32), tf.dtypes.cast(ZseqPassThresoldNotInt, tf.float32))	#zero all Vseq sectors (batchIndices, neurons) which pass threshold; prepare for addition
				VseqUpdated = SANItf2_algorithmSANIoperations.updateTensorCells(VseqExistingOld, tf.cast(ZseqPassThresold, tf.float32))	#tf.math.logical_or(Vseq[generateParameterNameSeq(l, s, "Vseq")], ZseqPassThresold)
				VseqUpdated = tf.cast(VseqUpdated, tf.bool)
				ZseqExistingOld = tf.multiply(tf.dtypes.cast(Zseq[generateParameterNameSeq(l, s, "Zseq")], tf.float32), tf.dtypes.cast(ZseqPassThresoldNotInt, tf.float32))	#zero all Zseq sectors (batchIndices, neurons) which pass threshold; prepare for addition
				ZseqUpdated = SANItf2_algorithmSANIoperations.updateTensorCells(ZseqExistingOld, tf.cast(ZseqPassThresold, tf.float32))	#tf.math.logical_or(Zseq[generateParameterNameSeq(l, s, "Zseq")], ZseqPassThresold) 
				ZseqUpdated = tf.cast(ZseqUpdated, tf.bool)
				AseqUpdated, _ = sequentialActivationFunction(ZseqUpdated)
				
				#update parameter storage;
				Zseq[generateParameterNameSeq(l, s, "Zseq")] = ZseqUpdated			
				Aseq[generateParameterNameSeq(l, s, "Aseq")] = AseqUpdated
				Vseq[generateParameterNameSeq(l, s, "Vseq")] = VseqUpdated

				if(resetSequentialInputsIfOnlyFirstInputValid):
					if(s > 0):
						sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_or(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], ZseqPassThresold)	#record that a higher sequential input was newly activated
										
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
				
			Z1 = ZseqLast
			A1 = AseqLast
											
			A[generateParameterName(l, "A")] = A1
			Z[generateParameterName(l, "Z")] = Z1
			
			if(enforceTcontiguityConstraints):
				TMax[generateParameterName(l, "TMax")] = TMaxSeqLast
				TMin[generateParameterName(l, "TMin")] = TMinSeqLast
				TMaxPrevLayer = TMax[generateParameterName(l, "TMax")]
				TMinPrevLayer = TMin[generateParameterName(l, "TMin")]
							
			AprevLayer = A[generateParameterName(l, "A")]

	
	#return tf.nn.softmax(Z1)
	#return tf.nn.sigmoid(Z1)	#binary classification
	
	ZlastLayer = Z[generateParameterName(numberOfLayers, "Z")]
	if(useLearningRuleBackpropagation):
		ZlastLayer = tf.dtypes.cast(ZlastLayer, tf.float32)
		pred = SANItf2_algorithmSANIoperations.generatePrediction(ZlastLayer, Whead, applySoftmax=(not vectorisedOutput))	
	else:
		if(enforceTcontiguityConstraints):
			ZlastLayer = TcontiguityCalculateOutput(ZlastLayer)
		pred = tf.math.reduce_any(ZlastLayer, axis=1)
	
	return pred
	
def sequentialActivationFunction(Zseq):
	ZseqPassThresold = Zseq	#binary	
	ZseqThresholded = Zseq	#binary	#Aseq
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
	printAverage(TMaxPrevLayer, "TMaxPrevLayer", 1)
	printAverage(TMinPrevLayer, "TMinPrevLayer", 1)

	return TMinPrevLayer, TMaxPrevLayer
	
def TcontiguityLayerInitialiseTemporaryVars(l, TMinPrevLayer, TMaxPrevLayer):
	TMinPrevLayerAll, TMaxPrevLayerAll = (None, None)

	#declare variables used across all sequential input of neuron
	#primary vars;
	if(l == 1):
		if(supportSkipLayers):
			TMaxPrevLayerAll = TMaxPrevLayer
			TMinPrevLayerAll = TMinPrevLayer
			#print(TMinPrevLayer)
			#print(TMaxPrevLayer)
			#print(TMinPrevLayerAll)
			#print(TMaxPrevLayerAll)	
	else:
		if(supportSkipLayers):
			TMaxPrevLayerAll = tf.concat([TMaxPrevLayerAll, TMaxPrevLayer], 1)
			TMinPrevLayerAll = tf.concat([TMinPrevLayerAll, TMinPrevLayer], 1)

	return TMinPrevLayerAll, TMaxPrevLayerAll

def TcontiguitySequentialInputInitialiseTemporaryVars(l, s, TMinPrevLayer, TMaxPrevLayer, TMinPrevLayerAll, TMaxPrevLayerAll):
	TMinSeqInput, TMaxSeqInput = (None, None)
	
	#identify (hypothetical) activation of neuron sequential input
	if(supportFullConnectivity):
		print("neuralNetworkPropagationSANI error: supportFullConnectivity incomplete")
	else:
		if(supportSkipLayers):
			CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
			CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)
			TMaxSeqInput = tf.gather(TMaxPrevLayerAll, CseqCrossLayer, axis=1)
			TMinSeqInput = tf.gather(TMinPrevLayerAll, CseqCrossLayer, axis=1)
		else:
			TMaxSeqInput = tf.gather(TMaxPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
			TMinSeqInput = tf.gather(TMinPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
							
	return TMinSeqInput, TMaxSeqInput


def TcontiguitySequentialInputConstrainAseqInput(l, s, AseqInput, TMinSeqInput, TMaxSeqInput):
	AseqInput, TMinSeqInputThresholded, TMaxSeqInputThresholded = (None, None, None)
	
	#ensure that T continguous constraint is met (T threshold AseqInput);
	#NO: take the max subinput pathway only (ie matrix mult but with max() rather than sum() for each dot product)
	if(s > 0):
		numberSubinputsPerSequentialInput = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)
		multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

		if(enforceTcontiguityBetweenSequentialInputs):
			#1. first select the AseqInput inputs that have TMinSeq[l, s] contiguous with TMaxSeq[l, s-1]:
			TMaxSeqPrevPlus1 = TMaxSeq[generateParameterNameSeq(l, s-1, "TMaxSeq")]+1
			TMinSeqPrev = TMinSeq[generateParameterNameSeq(l, s-1, "TMinSeq")]

			#printAverage(TMaxSeqPrevPlus1, "TMaxSeqPrevPlus1PRE", 3)	
			TMaxSeqPrevPlus1Tiled = tf.tile(tf.reshape(TMaxSeqPrevPlus1, [batchSize, 1, n_h[l]]), multiples)
			#printAverage(TMaxSeqPrevPlus1Tiled, "TMaxSeqPrevPlus1TiledPRE", 3)
			TMinSeqInputThreshold = tf.math.equal(TMinSeqInput, TMaxSeqPrevPlus1Tiled)
			#printAverage(TMinSeqInputThreshold, "TMinSeqInputThresholdPRE", 3)	
			AseqInputTthresholded = tf.math.logical_and(AseqInput, TMinSeqInputThreshold)

			TMinSeqInputThresholded = TMinSeqPrev	#note this is recording the min of the sequence, not the sequential input	#OLD: TMaxSeqPrevPlus1

			#printAverage(AseqInputTthresholded, "AseqInputTthresholdedPRE", 3)			

			AseqInput = AseqInputTthresholded

		if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
			TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
			AseqInputTthresholded = tf.math.logical_and(AseqInput, TMaxSeqInputThreshold)

			TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w

			AseqInput = AseqInputTthresholded

	else:
		if(resetSequentialInputsTContiguity):
			#only reset first sequential input if TMaxSeqInput > TMax[l]

			numberSubinputsPerSequentialInput = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)
			multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

			TMaxtiled = tf.tile(tf.reshape(TMax[generateParameterName(l, "TMax")], [batchSize, 1, n_h[l]]), multiples)
			VseqTiled = tf.tile(tf.reshape(Vseq[generateParameterNameSeq(l, s, "Vseq")], [batchSize, 1, n_h[l]]), multiples)

			ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(VseqTiled), tf.math.greater(TMaxSeqInput, tf.dtypes.cast(TMaxtiled, tf.int32)))
			AseqInputTthresholded = tf.math.logical_and(AseqInput, ZseqHypotheticalResetThreshold)

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
				AseqInputTthresholded = tf.math.logical_and(AseqInput, TMaxSeqInputThreshold)

				TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w

				AseqInput = AseqInputTthresholded
	return AseqInput, TMinSeqInputThresholded, TMaxSeqInputThresholded

def TcontiguityUpdateArrays(l, s, ZseqPassThresold, TMinSeqInputThresholded, TMaxSeqInputThresholded):
	ZseqPassThresoldInt = tf.dtypes.cast(ZseqPassThresold, tf.int32)
	ZseqPassThresoldNot = tf.math.logical_not(ZseqPassThresold)
	ZseqPassThresoldNotInt = tf.dtypes.cast(ZseqPassThresoldNot, tf.int32)

	TMaxSeqPassThresoldUpdatedValues = tf.multiply(ZseqPassThresoldInt, TMaxSeqInputThresholded)
	TMaxSeqExistingOld = tf.multiply(TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")], ZseqPassThresoldNotInt)
	TMaxSeqUpdated = SANItf2_algorithmSANIoperations.updateTensorCells(TMaxSeqExistingOld, TMaxSeqPassThresoldUpdatedValues)	#tf.add(TMaxSeqUpdated, TMaxSeqPassThresoldUpdatedValues)

	TMinSeqPassThresoldUpdatedValues = tf.multiply(ZseqPassThresoldInt, TMinSeqInputThresholded)
	TMinSeqExistingOld = tf.multiply(TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")], ZseqPassThresoldNotInt)
	TMinSeqUpdated = SANItf2_algorithmSANIoperations.updateTensorCells(TMinSeqExistingOld, TMinSeqPassThresoldUpdatedValues)	#tf.add(TMinSeqExistingOld, TMinSeqPassThresoldUpdatedValues)					

	#update parameter storage;
	TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = TMaxSeqUpdated
	TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = TMinSeqUpdated				
	#printAverage(TMaxSeqUpdated, "TMaxSeqUpdated", 3)
	#printAverage(TMinSeqUpdated, "TMinSeqUpdated", 3)

def TcontiguityCalculateOutput(ZlastLayer):
	if(enforceTcontiguityStartAndEndOfSequence):
		TMaxLastLayer = TMax[generateParameterName(numberOfLayers, "TMax")] 
		TMinLastLayer = TMin[generateParameterName(numberOfLayers, "TMin")]

		printAverage(ZlastLayer, "ZlastLayer", 1)
		printAverage(TMaxLastLayer, "TMaxLastLayer", 1)
		printAverage(TMinLastLayer, "TMinLastLayer", 1)

		#multiples = tf.constant([1,numberOfLayers], tf.int32)
		#numberOfWordsInSentenceBatchTiled = tf.tile(tf.reshape(numberOfWordsInSentenceBatch, [batchSize, n_h[numberOfLayers]]), multiples)
		multiples = tf.constant([1,n_h[numberOfLayers]], tf.int32)
		numberOfWordsInSentenceBatchTiled = tf.tile(tf.reshape(numberOfWordsInSentenceBatch, [batchSize, 1]), multiples)

		TMaxLastLayerThreshold = tf.math.equal(TMaxLastLayer, numberOfWordsInSentenceBatchTiled-1)
		printAverage(TMaxLastLayerThreshold, "TMaxLastLayerThreshold", 1)
		TMinLastLayerThreshold = tf.math.equal(TMinLastLayer, 0)
		printAverage(TMaxLastLayerThreshold, "TMaxLastLayerThreshold", 1)

		ZlastLayer = tf.math.logical_and(ZlastLayer, TMaxLastLayerThreshold)
		printAverage(ZlastLayer, "ZlastLayer", 1)
		ZlastLayer = tf.math.logical_and(ZlastLayer, TMinLastLayerThreshold)
		printAverage(ZlastLayer, "ZlastLayer", 1)
	return ZlastLayer
	
					
def recordActivitySequentialInput(l, s, ZseqPassThresold):
	if(recordNetworkWeights):
		if(recordSubInputsWeighted):
			#apply sequential verification matrix to AseqInput (rather than ZseqHypothetical) to record the precise subinputs that resulted in a sequential input being activated:

			numberSubinputsPerSequentialInput = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)
			multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

			VseqBoolTiled =  tf.tile(tf.reshape(VseqBool, [batchSize, 1, n_h[l]]), multiples)
			AseqInputVerifiedTemp = tf.math.logical_and(AseqInput, VseqBoolTiled)

			AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")] = tf.math.logical_or(AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")], AseqInputVerifiedTemp)			

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
					neuronNewlyActivated = ZseqPassThresold
					neuronNewlyActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(neuronNewlyActivated, tf.float32), axis=0)
					BR[generateParameterName(l, "BR")] = tf.add(BR[generateParameterName(l, "BR")], neuronNewlyActivatedBatchSummed)

		if(recordSequentialInputsWeighted):	
			sequentialInputNewlyActivated = ZseqPassThresold
			sequentialInputNewlyActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(sequentialInputNewlyActivated, tf.float32), axis=0)
			WR[generateParameterName(l, "WR")] = tf.add(WR[generateParameterName(l, "WR")][s], sequentialInputNewlyActivatedBatchSummed)


