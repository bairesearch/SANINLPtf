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
if(algorithmSANI == "repeatedModules"):
	pass

#end common SANItf2_algorithmSANI.py code


def neuralNetworkPropagation(x, networkIndex=None):
	return neuralNetworkPropagationSANI(x)
	
def neuralNetworkPropagationSANI(x):
	
	x = tf.dtypes.cast(x, tf.float32)
	
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
	#if(performFunctionOfSequentialInputsWeighted):	
		#W	(dim: numberOfSequentialInputs*n_h[l])
		
	#combination vars (multilayer);
	#if(supportSkipLayers):
		#AprevLayerAll	#output vector (dim: batchSize*(n_h[0]+n_h[1]+...n_h[L]))

	#combination vars (per layer);		
	#if(allowMultipleSubinputsPerSequentialInput):
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
	#else:
		#if(performFunctionOfSequentialInputsWeighted):
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
	#print("x = ", x)
	
	for l in range(1, numberOfLayers+1):
		
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
			tMin, tMid, tMax, tMinLayerAll, tMidLayerAll, tMaxLayerAll = TcontiguityLayerInitialiseTemporaryVars(l)
					
		#combination vars;
		if(enforceTcontiguityConstraints):
			tMidSeqSum = tf.zeros([batchSize, n_h[l]], tf.int32)
		if(allowMultipleSubinputsPerSequentialInput):
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
		else:
			if(performFunctionOfSequentialInputsWeighted):
				AseqInputWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)
			
		for s in range(numberOfSequentialInputs):
			
			if(printStatus):
				print("\t\ts = " + str(s))

			if(useSparseTensors):
				if(useMultipleSubinputsPerSequentialInput):
					numberSubinputsPerSequentialInput = SANItf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInputSparseTensors(l, s)

			#calculate validation matrix based upon sequentiality requirements
			if(s == 0):
				if(performIndependentSubInputValidation):
					Vseq = tf.fill([batchSize, numberSubinputsPerSequentialInput, n_h[l]], True)
				else:
					Vseq = tf.fill([batchSize, n_h[l]], True)	#all values of Vseq0_l are always set to 1 as they have no sequential dependencies		
			else:
				if(performIndependentSubInputValidation):
					multiples = tf.constant([1, numberSubinputsPerSequentialInput, 1], tf.int32) 
					VseqPrevTest = tf.tile(tf.reshape(VseqPrev, [batchSize, 1, n_h[l]]), multiples)
				else:
					VseqPrevTest = VseqPrev
	
			if(enforceTcontiguityConstraints):
				tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest = TcontiguitySequentialInputInitialiseTemporaryVars(l, s, tMinLayerAll, tMidLayerAll, tMaxLayerAll, tMin, tMid, tMax, tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev)
				
			if(s > 0):
				if(enforceTcontiguityConstraints):
					Vseq = TcontiguitySequentialInputInitialiseValidationMatrix(l, s, VseqPrevTest, tMinSeqTest, tMidSeqTest, tMaxSeqTest)
				else:
					Vseq = VseqPrevTest	#if previous sequentiality check fails, then all future sequentiality checks must fail
				
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
					multiples = tf.constant([1, numberSubinputsPerSequentialInput, 1], tf.int32)
					VseqFloatTiled = tf.tile(tf.reshape(VseqFloat, [batchSize, 1, n_h[l]]), multiples)
					AseqInput = tf.multiply(VseqFloatTiled, AseqInput)
			else:
				AseqInput = tf.multiply(VseqFloat, AseqInput)

			if(performFunctionOfSequentialInputsWeighted):
				multiples = tf.constant([batchSize,1], tf.int32)
				Wtiled = tf.tile(tf.reshape(W[generateParameterName(l, "W")][s], [1, n_h[l]]), multiples)
					
			ZseqIndex = None	#initialise						
			if(allowMultipleSubinputsPerSequentialInput):
				if(useSparseTensors):
					if(performTindependentFunctionOfSubInputs):
						if(performFunctionOfSubInputsWeighted):
							multiplesSeq = tf.constant([batchSize,1,1], tf.int32)
							WseqTiled = tf.tile(tf.reshape(Wseq[generateParameterNameSeq(l, s, "Wseq")], [1, numberSubinputsPerSequentialInput, n_h[l]]), multiplesSeq)
							AseqInputWeighted = tf.multiply(AseqInput, WseqTiled)
						else:
							AseqInputWeighted = AseqInput

						if(performFunctionOfSubInputsSummation):
							Zseq = tf.math.reduce_sum(AseqInputWeighted, axis=1)
						elif(performFunctionOfSubInputsMax):
							#take sub input with max input signal*weight
							Zseq = tf.math.reduce_max(AseqInputWeighted, axis=1)
							ZseqIndex = tf.math.argmax(AseqInputWeighted, axis=1)
					else:
						if(performFunctionOfSubInputsWeighted):
							AseqInputReshaped = AseqInput	#(batchSize, previousLayerFeaturesUsed, layerNeurons)
							AseqInputReshaped = tf.transpose(AseqInputReshaped, [2, 0, 1])	#layerNeurons, batchSize, previousLayerFeaturesUsed
							WseqReshaped = Wseq[generateParameterNameSeq(l, s, "Wseq")]	#previousLayerFeaturesUsed, layerNeurons
							WseqReshaped = tf.transpose(WseqReshaped, [1, 0])	#layerNeurons, previousLayerFeaturesUsed
							WseqReshaped = tf.expand_dims(WseqReshaped, 2)	#layerNeurons, previousLayerFeaturesUsed, 1
							ZseqHypothetical = tf.matmul(AseqInputReshaped, WseqReshaped)	#layerNeurons, batchSize, 1
							ZseqHypothetical = tf.squeeze(ZseqHypothetical, axis=2)	#layerNeurons, batchSize
							ZseqHypothetical = tf.transpose(ZseqHypothetical, [1, 0])	#batchSize, layerNeurons
							ZseqHypothetical = tf.add(ZseqHypothetical, Bseq[generateParameterNameSeq(l, s, "Bseq")])	#batchSize, layerNeurons
							Zseq = ZseqHypothetical
						else:
							print("SANItf2_algorithmSANIrepeatedModules error: allowMultipleSubinputsPerSequentialInput:useSparseTensors:!performTindependentFunctionOfSubInputs:!performFunctionOfSubInputsWeighted")
							exit(0)
				else:
					if(performFunctionOfSubInputsWeighted):
						ZseqHypothetical = tf.add(tf.matmul(AseqInput, Wseq[generateParameterNameSeq(l, s, "Wseq")]), Bseq[generateParameterNameSeq(l, s, "Bseq")])
						Zseq = ZseqHypothetical
					elif(performFunctionOfSubInputsAverage):
						AseqInputAverage = tf.math.reduce_mean(AseqInput, axis=1)	#take average
						multiples = tf.constant([1,n_h[l]], tf.int32)
						ZseqHypothetical = tf.tile(tf.reshape(AseqInputAverage, [batchSize, 1]), multiples)		
						Zseq = ZseqHypothetical				
					
				Aseq, ZseqPassThresold = sequentialActivationFunction(Zseq)
				
				if(performFunctionOfSequentialInputs):
					#these are all used for different methods of sequential input summation
					if(performFunctionOfSubInputsNonlinear):
						AseqSum = tf.add(AseqSum, Aseq)
					else:
						ZseqSum = tf.add(ZseqSum, Zseq)
					if(performFunctionOfSequentialInputsWeighted):
						#apply weights to input of neuron sequential input
						if(performFunctionOfSubInputsNonlinear):
							AseqWeighted = tf.multiply(Aseq, Wtiled)
							AseqWeightedSum = tf.math.add(AseqWeightedSum, AseqWeighted)	
						else:
							ZseqWeighted = tf.multiply(Zseq, Wtiled)
							ZseqWeightedSum = tf.math.add(ZseqWeightedSum, ZseqWeighted)
			else:
				Zseq = AseqInput	
				Aseq, ZseqPassThresold = sequentialActivationFunction(Zseq)
				if(performFunctionOfSequentialInputsWeighted):
					AseqInputWeighted = tf.multiply(AseqInput, Wtiled)
					AseqInputWeightedSum = tf.add(AseqInputWeightedSum, AseqInputWeighted)
			

			if(enforceTcontiguityConstraints):
				tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev = TcontiguitySequentialInputFinaliseTemporaryVars(l, s, tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest, ZseqIndex)		
	
			#if(performTindependentFunctionOfSubInputs):
			if(performIndependentSubInputValidation):
				if(performFunctionOfSubInputsSummation):
					VseqReduced = tf.reduce_any(Vseq, axis=1)
				elif(performFunctionOfSubInputsMax):
					#take subinput with max input signal (AseqInput)
					VseqReduced = Vseq[:, ZseqIndex, :]	
			else:
				VseqReduced = Vseq
	
			#calculate updated Vseq
			VseqUpdated = tf.math.logical_or(VseqReduced, ZseqPassThresold)	#OLD: VseqUpdated = Vseq
			VseqPrev = VseqUpdated
			
			if(s == numberOfSequentialInputs-1):
				VseqLast = VseqPrev
			
		#calculate A (output) matrix of current layer		
		if(allowMultipleSubinputsPerSequentialInput):
			if(performFunctionOfSequentialInputs):
				if(performFunctionOfSequentialInputsWeighted):
					if(performFunctionOfSubInputsNonlinear):
						Z = AseqWeightedSum			
					else:
						Z = ZseqWeightedSum			
				else:
					if(performFunctionOfSubInputsNonlinear):
						Z = AseqSum			
					else:
						Z = ZseqSum
				if(sequentialInputCombinationModeSummationAveraged):
					Z = Z/numberOfSequentialInputs
				if(performFunctionOfSequentialInputsNonlinear):
					A = activationFunction(Z)
				else:
					A = Z
				if(performFunctionOfSequentialInputsVerify):
					Z = tf.multiply(Z, tf.dtypes.cast(VseqLast, tf.float32))
					A = tf.multiply(A, tf.dtypes.cast(VseqLast, tf.float32))
					
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
				ZseqLast = Zseq
				AseqLast = Aseq
				#VseqLastFloat = VseqFloat
				Z = ZseqLast
				A = AseqLast
		else:
			if(performFunctionOfSequentialInputsWeighted):
				#Z = tf.add(tf.matmul(AseqAll, W[generateParameterName(l, "W")]), B[generateParameterName(l, "B")])
				Z = AseqInputWeightedSum
				if(performFunctionOfSequentialInputsNonlinear):
					A = activationFunction(Z)
				else:
					A = Z
			else:
				#CHECKTHIS;
				ZseqLast = Zseq
				AseqLast = Aseq
				Z = ZseqLast
				A = AseqLast
				
		AprevLayer = A

	if(useLearningRuleBackpropagation):
		pred = SANItf2_algorithmSANIoperations.generatePrediction(Z, Whead, applySoftmax=(not vectorisedOutput))
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



def TcontiguitySequentialInputInitialiseTemporaryVars(l, s, tMinLayerAll, tMidLayerAll, tMaxLayerAll, tMin, tMid, tMax, tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev):

	tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest = (None, None, None, None, None, None, None)

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
			if(enforceTcontiguityConstraints):
				tMinSeqPrevTest = tMinSeqPrev
				tMidSeqPrevTest = tMidSeqPrev
				tMaxSeqPrevTest = tMaxSeqPrev


	return tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest



def TcontiguitySequentialInputInitialiseValidationMatrix(l, s, VseqPrevTest, tMinSeqTest, tMidSeqTest, tMaxSeqTest):

	VseqTconstrained = None
	
	#calculate validation matrix based upon sequentiality requirements
	if(s > 0):
		if(sequentialityMode == "default"):
			#the first sub input of sequential input #2 must fire after the last subinput of sequential input #1
			VseqTconstrained = tf.math.greater(tMinSeqTest, tMaxSeqPrevTest)
		elif(sequentialityMode == "temporalCrossoverAllowed"):
			#the last sub input of sequential input #1 can fire after the first subinput of sequential input #2
			VseqTconstrained = tf.math.greater(tMaxSeqTest, tMaxSeqPrevTest)
		elif(sequentialityMode == "contiguousInputEnforced"):
			#the last sub input of sequential input #1 must fire immediately before the first subinput of sequentialInput #2
			VseqTconstrained = tf.math.equal(tMinSeqTest, tMaxSeqPrevTest+1)	#TODO: verify that the +1 here gets properly broadcasted	
		VseqTconstrained = tf.math.logical_and(VseqTconstrained, VseqPrevTest)	#if previous sequentiality check fails, then all future sequentiality checks must fail
	else:
		print("TcontiguitySequentialInputInitialiseTemporaryVars2 error; requires s > 0")
		exit()
		
	return VseqTconstrained


def TcontiguitySequentialInputFinaliseTemporaryVars(l, s, tMinSeq, tMidSeq, tMaxSeq, tMinSeqTest, tMidSeqTest, tMaxSeqTest, ZseqIndex):

	tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev = (None, None, None)
	
	#generate reduced versions of tMin/Mid/MaxSeq with only sequentially valid elements
	if(performIndependentSubInputValidation):
		if(performFunctionOfSubInputsSummation):
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
		else:
			#take subinput with max input signal (AseqInput)
			tMinSeqValidatedReduced = tMinSeq[:, ZseqIndex, :]
			tMidSeqValidatedReduced = tMidSeq[:, ZseqIndex, :]
			tMaxSeqValidatedReduced = tMaxSeq[:, ZseqIndex, :]

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
	else:
		tMinSeqPrev = tMinSeqTest
		tMidSeqPrev = tMidSeqTest
		tMaxSeqPrev = tMaxSeqTest
	
	return tMinSeqPrev, tMidSeqPrev, tMaxSeqPrev




