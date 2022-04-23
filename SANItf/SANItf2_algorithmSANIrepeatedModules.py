"""SANItf2_algorithmSANIrepeatedModules.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SANItf2.py

# Usage:
see SANItf2.py

# Description:
SANItf algorithm SANI repeated modules - define Sequentially Activated Neuronal Input neural network with repeated modules

Neural modules cannot be shared between different areas of input sequence.

"""

#start common SANItf2_algorithmSANI.py code:

import tensorflow as tf
import numpy as np
from ANNtf2_operations import * #generateParameterNameSeq, generateParameterName
import SANItf2_algorithmSANIoperations
from SANItf2_algorithmSANIglobalDefs import *
import ANNtf2_globalDefs


#parameters
#static parameters (convert from tf.variable to tf.constant?):
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
#static parameters (convert from tf.variable to tf.constant?):
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
#	#variable parameters (tf.variable): 
#	if(allowMultipleSubinputsPerSequentialInput):
#		if(performSummationOfSubInputsWeighted):
#			Wseq = {}	#weights matrix
#			Bseq = {}	#biases vector
#	if(performSummationOfSequentialInputsWeighted):
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

def defineNetworkParametersSANIwrapper(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, useSmallSentenceLengths, numberOfFeaturesPerWord):
	global n_h
	global numberOfLayers
	n_h, numberOfLayers = SANItf2_algorithmSANIoperations.defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, useSmallSentenceLengths, numberOfFeaturesPerWord)
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
if(algorithmSANI == "repeatedModules"):
	pass

#end common SANItf2_algorithmSANI.py code


def neuralNetworkPropagation(x, networkIndex=None):
	return neuralNetworkPropagationSANI(x)
	
def neuralNetworkPropagationSANI(x):
		
	batchSize = x.shape[0]

	#note connectivity indexes are used rather than sparse weight matrices due to limitations in current tf2 sparse tensor implementation
	
	#definitions for reference:
	
	#neuron sequential input vars;
	#x/AprevLayer	#output vector (dim: batchSize*n_h[l])
	#if(allowMultipleSubinputsPerSequentialInput):
		#Cseq	#static connectivity matrix (int) - indexes of neurons on prior layer stored; mapped to W  (dim: numberSubinputsPerSequentialInput*n_h[l])
		#if(supportSkipLayers):
			#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W (dim: numberSubinputsPerSequentialInput*n_h[l])
		#Wseq #weights of connections; see Cseq (dim: numberSubinputsPerSequentialInput*n_h[l])
		#AseqSum	#combination variable
	#else:
		#Cseq	#static connectivity vector (int) - indexes of neurons on prior layer stored; mapped to W - defines which prior layer neuron a sequential input is connected to (dim: n_h[l])
		#if(supportSkipLayers):
			#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W - defines which prior layer a sequential input is connected to  (dim: n_h[l])
		#Wseq #weights of connections; see Cseq (dim: n_h[l])
	#if(performIndependentSubInputValidation):
		#Vseq	#mutable verification vector (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l] - regenerated for each sequential input index)
	#else
		#Vseq	#mutable verification vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)
	#Zseq	#neuron activation function input vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)
	#Aseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)
	
	#neuron vars;
	#Q
	#Z	#neuron activation function input (dim: batchSize*n_h[l])
	#A	#neuron activation function output (dim: batchSize*n_h[l])
	#if(performSummationOfSequentialInputsWeighted):	
		#W	(dim: numberOfSequentialInputs*n_h[l])
		
	#combination vars (multilayer);
	#if(supportSkipLayers):
		#AprevLayerAll	#output vector (dim: batchSize*(n_h[0]+n_h[1]+...n_h[L]))

	#combination vars (per layer);		
	#if(allowMultipleSubinputsPerSequentialInput):
		#if(performSummationOfSequentialInputs):
			#these are all used for different methods of sequential input summation:
			#if(performSummationOfSequentialInputsWeighted):
				#if(performSummationOfSubInputsNonlinear):
					#AseqWeightedSum	#(dim: batchSize*n_h[l])
				#else:
					#ZseqWeightedSum	#(dim: batchSize*n_h[l])
			#else:
				#if(performSummationOfSubInputsNonlinear):
					#AseqSum	#(dim: batchSize*n_h[l])
				#else:
					#ZseqSum	#(dim: batchSize*n_h[l])
	#else:
		#if(performSummationOfSequentialInputsWeighted):
			#AseqInputWeightedSum	#(dim: batchSize*n_h[l])	#aka ZseqWeightedSum

	#Tcontiguity vars;
	#if(enforceTcontiguityConstraints):
		#neuron sequential input vars;
		#if(performIndependentSubInputValidation):
			#tMinSeq	#mutable time vector (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l])
			#tMidSeq	#mutable time vector (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l])
			#tMaxSeq	#mutable time vector (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l])
		#else
			#tMinSeq	#mutable time vector (dim: batchSize*n_h[l])
			#tMidSeq	#mutable time vector (dim: batchSize*n_h[l])
			#tMaxSeq	#mutable time vector (dim: batchSize*n_h[l])
		#neuron vars;
		#tMin	#mutable time vector (dim: batchSize*n_h[l-1])
		#tMid	#mutable time vector (dim: batchSize*n_h[l-1])
		#tMax	#mutable time vector (dim: batchSize*n_h[l-1])	
		#combination vars (multilayer);
		#if(supportSkipLayers):
			#tMinLayerAll	#mutable time vector (dim: batchSize*(n_h[0]+n_h[1]+...n_h[L]))
			#tMidLayerAll	#mutable time vector (dim: batchSize*(n_h[0]+n_h[1]+...n_h[L]))
			#tMaxLayerAll	#mutable time vector (dim: batchSize*(n_h[0]+n_h[1]+...n_h[L]))
		#combination vars (per layer);
		#tMidSeqSum	#(dim: batchSize*n_h[l])
				
			
	AprevLayer = x
	
	for l in range(1, numberOfLayers+1):
		
		if(printStatus):
			print("\tl = " + str(l))
		
		#declare variables used across all sequential input of neuron
		#primary vars;
		if(l == 1):
			if(supportSkipLayers):
				AprevLayerAll = AprevLayer	#x
		else:
			if(supportSkipLayers):
				AprevLayerAll = tf.concat([AprevLayerAll, AprevLayer], 1)
		if(enforceTcontiguityConstraints):
			tMin, tMid, tMax, tMinLayerAll, tMidLayerAll, tMaxLayerAll = TcontiguityLayerInitialiseTemporaryVars(l)
					
		#combination vars;
		if(enforceTcontiguityConstraints):
			tMidSeqSum = tf.zeros([batchSize, n_h[l]], tf.int32)
		if(allowMultipleSubinputsPerSequentialInput):
			if(performSummationOfSequentialInputs):
				#these are all used for different methods of sequential input summation:
				if(performSummationOfSequentialInputsWeighted):
					if(performSummationOfSubInputsNonlinear):
						AseqWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)
					else:
						ZseqWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)
				else:
					if(performSummationOfSubInputsNonlinear):
						AseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)
					else:
						ZseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)
		else:
			if(performSummationOfSequentialInputsWeighted):
				AseqInputWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)
			
		for s in range(numberOfSequentialInputs):
			
			if(printStatus):
				print("\t\ts = " + str(s))

			#calculate validation matrix based upon sequentiality requirements
			if(s == 0):
				if(performIndependentSubInputValidation):
					Vseq = tf.fill([batchSize, numberSubinputsPerSequentialInput, n_h[l]], True)
				else:
					Vseq = tf.fill([batchSize, n_h[l]], True)	#all values of Vseq0_l are always set to 1 as they have no sequential dependencies		
			else:
				if(performIndependentSubInputValidation):
					multiples = tf.constant([1, numberSubinputsPerSequentialInput, 1], tf.int32) 
					#printShape(tMinSeqPrev, "tMinSeqPrev")
					VseqPrevTest = tf.tile(tf.reshape(VseqPrev, [batchSize, 1, n_h[l]]), multiples)
				else:
					VseqPrevTest = VseqPrev
					
			#print("tsLxSx2" + str(tf.timestamp(name="tsLxSx2")))
			if(enforceTcontiguityConstraints):
				VseqUpdated, tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest = TcontiguitySequentialInputInitialiseTemporaryVars(l, s, VseqPrevTest,  tMinLayerAll, tMidLayerAll, tMaxLayerAll, tMin, tMid, tMax, tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev)
			else:
				if(s > 0):
					VseqUpdated = VseqPrevTest	#if previous sequentiality check fails, then all future sequentiality checks must fail
			if(s > 0):
				Vseq = VseqUpdated
					
			VseqInt = tf.dtypes.cast(Vseq, tf.int32)
			VseqFloat = tf.dtypes.cast(VseqInt, tf.float32)
			
			#identify input of neuron sequential input
			if(supportSkipLayers):
				CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
				CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)
				AseqInput = tf.gather(AprevLayerAll, CseqCrossLayer, axis=1)
			else:
				AseqInput = tf.gather(AprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
				
			
			#apply sequential validation matrix
			if(allowMultipleSubinputsPerSequentialInput):
				if(performIndependentSubInputValidation):
					AseqInput = tf.multiply(VseqFloat, AseqInput)
				else:
					#checkthis:
					multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
					VseqFloatTiled = tf.tile(tf.reshape(VseqFloat, [batchSize, 1, n_h[l]]), multiples)
					AseqInput = tf.multiply(VseqFloatTiled, AseqInput)
			else:
				AseqInput = tf.multiply(VseqFloat, AseqInput)

			if(performSummationOfSequentialInputsWeighted):
				multiples = tf.constant([batchSize,1], tf.int32)
				Wtiled = tf.tile(tf.reshape(W[generateParameterName(l, "W")][s], [1, n_h[l]]), multiples)
											
			if(allowMultipleSubinputsPerSequentialInput):
				if(performSummationOfSubInputsWeighted):
					multiplesSeq = tf.constant([batchSize,1,1], tf.int32)
					WseqTiled = tf.tile(tf.reshape(Wseq[generateParameterNameSeq(l, s, "Wseq")], [1, numberSubinputsPerSequentialInput, n_h[l]]), multiplesSeq)
					AseqInputWeighted = tf.multiply(AseqInput, WseqTiled)
				else:
					AseqInputWeighted = AseqInput
				
				if(performSummationOfSubInputs):
					Zseq = tf.math.reduce_sum(AseqInputWeighted, axis=1)
				else:
					#take sub input with max input signal*weight
					Zseq = tf.math.reduce_max(AseqInputWeighted, axis=1)
					ZseqIndex = tf.math.argmax(AseqInputWeighted, axis=1)
					
				Aseq, _ = activationFunction(Zseq)
				
				if(performSummationOfSequentialInputs):
					#these are all used for different methods of sequential input summation
					if(performSummationOfSubInputsNonlinear):
						AseqSum = tf.add(AseqSum, Aseq)
					else:
						ZseqSum = tf.add(ZseqSum, Zseq)
					if(performSummationOfSequentialInputsWeighted):
						#apply weights to input of neuron sequential input
						if(performSummationOfSubInputsNonlinear):
							AseqWeighted = tf.multiply(Aseq[generateParameterNameSeq(l, s, "Aseq")], Wtiled)
							AseqWeightedSum = tf.math.add(AseqWeightedSum, AseqWeighted)	
						else:
							ZseqWeighted = tf.multiply(Zseq[generateParameterNameSeq(l, s, "Zseq")], Wtiled)
							ZseqWeightedSum = tf.math.add(ZseqWeightedSum, ZseqWeighted)
			else:
				if(performSummationOfSequentialInputsWeighted):
					AseqInputWeighted = tf.multiply(AseqInput, Wtiled)
					AseqInputWeightedSum = tf.add(AseqInputWeightedSum, AseqInputWeighted)
				else:
					#CHECKTHIS;
					Zseq = AseqInput	
					Aseq, _ = activationFunction(Zseq)
			
			if(enforceTcontiguityConstraints):
				VseqPrev, tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev = TcontiguitySequentialInputFinaliseTemporaryVars(l, s, tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest)
			else:
				VseqPrev = Vseq
			
		#calculate A (output) matrix of current layer		
		if(allowMultipleSubinputsPerSequentialInput):
			if(performSummationOfSequentialInputs):	
				if(performSummationOfSequentialInputs):
					if(performSummationOfSequentialInputsWeighted):
						if(performSummationOfSubInputsNonlinear):
							Z1 = AseqWeightedSum			
						else:
							Z1 = ZseqWeightedSum			
					else:
						if(performSummationOfSubInputsNonlinear):
							Z1 = AseqSum			
						else:
							Z1 = ZseqSum
					if(sequentialInputCombinationModeSummationAveraged):
						Z1 = Z1/numberOfSequentialInputs
					if(performSummationOfSequentialInputsNonlinear):
						A1 = activationFunction(Z1)
					else:
						A1 = Z1
				if(performSummationOfSequentialInputsVerify):
					Z = tf.multiply(Z, tf.dtypes.cast(VseqLast, tf.float32))
					A = tf.multiply(A, tf.dtypes.cast(VseqLast, tf.float32))
			else:
				ZseqLast = Zseq
				AseqLast = Aseq
				#VseqLastFloat = VseqFloat
				Z = ZseqLast
				A = AseqLast
		else:
			if(performSummationOfSequentialInputsWeighted):
				#Z = tf.add(tf.matmul(AseqAll, W[generateParameterName(l, "W")]), B[generateParameterName(l, "B")])
				Z = AseqInputWeightedSum
				if(performSummationOfSequentialInputsNonlinear):
					A = activationFunction(Z)
				else:
					A = Z
			else:
				#CHECKTHIS;
				ZseqLast = Zseq
				AseqLast = Aseq
				#VseqLastFloat = VseqFloat
				Z = ZseqLast
				A = AseqLast
				
		AprevLayer = A

	if(useLearningRuleBackpropagation):
		pred = SANItf2_algorithmSANIoperations.generatePrediction(Z, Whead)
	else:
		pred = tf.nn.softmax(Z)
	
	return pred

def sequentialActivationFunction(Zseq):
	#threshold output/check output threshold
	if(performThresholdOfSubInputsNonlinear):
		ZseqThresholded = activationFunction(Zseq)
		ZseqPassThresold = tf.math.greater(ZseqThresholded, 0.0)
	else:
		ZseqPassThresold = tf.math.greater(Zseq, sequentialInputActivationThreshold)
		ZseqThresholded = tf.multiply(Zseq, tf.dtypes.cast(ZseqPassThresold, tf.float32))	

	return ZseqThresholded, ZseqPassThresold	
	
def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A				

 
def TcontiguityLayerInitialiseTemporaryVars(l):
	tMin, tMid, tMax, tMinLayerAll, tMidLayerAll, tMaxLayerAll = (None, None, None, None, None, None)

	#declare variables used across all sequential input of neuron
	#primary vars;
	if(l == 1):
		tMinL0Row = tf.range(0, n_h[l-1], delta=1, dtype=tf.int32)	#n_h[l-1] = datasetNumFeatures
		tMidL0Row = tf.range(0, n_h[l-1], delta=1, dtype=tf.int32)
		tMaxL0Row = tf.range(0, n_h[l-1], delta=1, dtype=tf.int32)
		multiples = tf.constant([batchSize, 1], tf.int32) 
		tMin = tf.tile(tf.reshape(tMinL0Row, [1, n_h[l-1]]), multiples)
		tMid = tf.tile(tf.reshape(tMidL0Row, [1, n_h[l-1]]), multiples)
		tMax = tf.tile(tf.reshape(tMaxL0Row, [1, n_h[l-1]]), multiples)
		if(supportSkipLayers):
			tMinLayerAll = tMin
			tMidLayerAll = tMid
			tMaxLayerAll = tMax
	else:
		tMin = tMinNext
		tMid = tMidNext
		tMax = tMaxNext
		if(supportSkipLayers):
			tMinLayerAll = tf.concat([tMinLayerAll, tMin], 1)
			tMidLayerAll = tf.concat([tMidLayerAll, tMid], 1)
			tMaxLayerAll = tf.concat([tMaxLayerAll, tMax], 1)
				
	return tMin, tMid, tMax, tMinLayerAll, tMidLayerAll, tMaxLayerAll

def TcontiguitySequentialInputInitialiseTemporaryVars(l, s, VseqPrevTest, tMinLayerAll, tMidLayerAll, tMaxLayerAll, tMin, tMid, tMax, tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev):

	VseqUpdated, tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest = (None, None, None, None, None, None, None)

	#calculate tMin/Mid/Max for sequential input
	if(supportFullConnectivity):
		print("neuralNetworkPropagationSANI error: supportFullConnectivity incomplete")	
	else:
		#print("tsLxSx1" + str(tf.timestamp(name="tsLxSx1")))
		if(supportSkipLayers):
			CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
			CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)
			tMinSeq = tf.gather(tMinLayerAll, CseqCrossLayer, axis=1)
			tMidSeq = tf.gather(tMidLayerAll, CseqCrossLayer, axis=1)
			tMaxSeq = tf.gather(tMaxLayerAll, CseqCrossLayer, axis=1)	
		else:
			tMinSeq = tf.gather(tMin, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
			tMidSeq = tf.gather(tMid, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
			tMaxSeq = tf.gather(tMax, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
		tMinSeqTest = tMinSeq
		tMidSeqTest = tMidSeq
		tMaxSeqTest = tMaxSeq
		if(allowMultipleSubinputsPerSequentialInput):
			if not(performIndependentSubInputValidation):
				tMinSeqReduced = tf.math.reduce_min(tMinSeq, axis=1)
				tMidSeqReduced = tf.math.reduce_mean(tMidSeq, axis=1)
				tMaxSeqReduced = tf.math.reduce_max(tMaxSeq, axis=1)
				tMinSeqTest = tMinSeqReduced
				tMidSeqTest = tMidSeqReduced
				tMaxSeqTest = tMaxSeqReduced

	#calculate validation matrix based upon sequentiality requirements
	if(s > 0):
		if(performIndependentSubInputValidation):
			multiples = tf.constant([1, numberSubinputsPerSequentialInput, 1], tf.int32) 
			if(enforceTcontiguityConstraints):
				tMinSeqPrevTest = tf.tile(tf.reshape(tMinSeqPrev, [batchSize, 1, n_h[l]]), multiples)
				tMidSeqPrevTest = tf.tile(tf.reshape(tMidSeqPrev, [batchSize, 1, n_h[l]]), multiples)
				tMaxSeqPrevTest = tf.tile(tf.reshape(tMaxSeqPrev, [batchSize, 1, n_h[l]]), multiples)
		else:
			VseqPrevTest = VseqPrev
			if(enforceTcontiguityConstraints):
				tMinSeqPrevTest = tMinSeqPrev
				tMidSeqPrevTest = tMidSeqPrev
				tMaxSeqPrevTest = tMaxSeqPrev

		if(sequentialityMode == "default"):
			#the first sub input of sequential input #2 must fire after the last subinput of sequential input #1
			VseqUpdated = tf.math.greater(tMinSeqTest, tMaxSeqPrevTest)
		elif(sequentialityMode == "temporalCrossoverAllowed"):
			#the last sub input of sequential input #1 can fire after the first subinput of sequential input #2
			VseqUpdated = tf.math.greater(tMaxSeqTest, tMaxSeqPrevTest)
		elif(sequentialityMode == "contiguousInputEnforced"):
			#the last sub input of sequential input #1 must fire immediately before the first subinput of sequentialInput #2
			VseqUpdated = tf.math.equal(tMinSeqTest, tMaxSeqPrevTest+1)	#TODO: verify that the +1 here gets properly broadcasted	
		VseqUpdated = tf.math.logical_and(VseqUpdated, VseqPrevTest)	#if previous sequentiality check fails, then all future sequentiality checks must fail

	return VseqUpdated, tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest


def TcontiguitySequentialInputFinaliseTemporaryVars(l, s, tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest):

	VseqPrev, tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev = (None, None, None, None)
	
	#generate reduced versions of tMin/Mid/MaxSeq with only sequentially valid elements
	if(performIndependentSubInputValidation):
		if(performSummationOfSubInputs):
			#mask tMin/Mid/MaxSeq based on sequentiality validation matrix 
			VseqNot = tf.logical_not(Vseq)  
			VseqNotInt = tf.dtypes.cast(VseqNot, tf.int32)
			VseqSumAxis1 = tf.math.reduce_sum(VseqInt, axis=1)
			VseqIntMin = tf.add(tf.multiply(VseqNotInt, veryLargeInt), 1)  	#if VseqInt[x] = 0/False, then VseqIntMin = veryLargeInt. If VseqInt[x] = 1/True, then VseqIntMin = 1
			VseqIntMid = tf.multiply(VseqInt, 1)				#if VseqInt[x] = 0/False, then VseqIntMid = 0. If VseqInt[x] = 1/True, then VseqIntMid = 1
			VseqIntMax = tf.multiply(VseqInt, 1)				#if VseqInt[x] = 0/False, then VseqIntMax = 0. If VseqInt[x] = 1/True, then VseqIntMax = 1
			tMinSeqOnlyValid = tf.multiply(tMinSeq, VseqIntMin)
			tMidSeqOnlyValid = tf.multiply(tMidSeq, VseqIntMid)
			tMaxSeqOnlyValid = tf.multiply(tMaxSeq, VseqIntMax)
			tMinSeqValidatedReduced = tf.math.reduce_min(tMinSeqOnlyValid, axis=1)
			tMidSeqValidatedReduced = tf.divide(tf.math.reduce_sum(tMidSeqOnlyValid, axis=1), VseqSumAxis1)
			tMaxSeqValidatedReduced = tf.math.reduce_max(tMaxSeqOnlyValid, axis=1)
			tMinSeqValidatedReduced = tf.dtypes.cast(tMinSeqValidatedReduced, tf.int32)
			tMidSeqValidatedReduced = tf.dtypes.cast(tMidSeqValidatedReduced, tf.int32)
			tMaxSeqValidatedReduced = tf.dtypes.cast(tMaxSeqValidatedReduced, tf.int32)
			VseqReduced = tf.reduce_any(Vseq, axis=1)
		else:
			#take subinput with max input signal (AseqInput)
			tMinSeqValidatedReduced = tMinSeq[:, ZseqIndex, :]
			tMidSeqValidatedReduced = tMidSeq[:, ZseqIndex, :]
			tMaxSeqValidatedReduced = tMaxSeq[:, ZseqIndex, :]
			VseqReduced = Vseq[:, ZseqIndex, :]

	#calculate tMin/Mid/MaxNext (ie the tMin/Mid/Max values to be assigned to the current layer after it has been processed):
	if(s == 0):
		if(performIndependentSubInputValidation):
			tMinSeqFirst = tMinSeqValidatedReduced
		else:
			tMinSeqFirst = tMinSeqTest
	if(s == numberOfSequentialInputs-1):
		if(performIndependentSubInputValidation):
			tMaxSeqLast = tMaxSeqValidatedReduced
		else:
			tMaxSeqLast = tMaxSeqTest
	if(performIndependentSubInputValidation):
		tMidSeqSum = tf.math.add(tMidSeqSum, tMidSeqValidatedReduced)
	else:
		tMidSeqSum = tf.math.add(tMidSeqSum, tMidSeqTest)

	if(s == numberOfSequentialInputs-1):
		tMinNext = tMinSeqFirst
		tMidNext = tf.dtypes.cast(tf.math.divide(tMidSeqSum, s+1), tf.int32)
		tMaxNext = tMaxSeqLast			

	if(performIndependentSubInputValidation):
		tMinSeqPrev = tMinSeqValidatedReduced
		tMidSeqPrev = tMidSeqValidatedReduced
		tMaxSeqPrev = tMaxSeqValidatedReduced
		VseqPrev = VseqReduced
	else:
		tMinSeqPrev = tMinSeqTest
		tMidSeqPrev = tMidSeqTest
		tMaxSeqPrev = tMaxSeqTest
		VseqPrev = Vseq
	
	return VseqPrev, tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev
