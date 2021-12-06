"""ANNtf2_algorithmLIANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm LIANN - define local inhibition artificial neural network (force neural independence)

Emulates unsupervised singular value decomposition (SVD/factor analysis) learning for all hidden layers

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import copy

#select learningAlgorithm:
learningAlgorithmStochastic = False
learningAlgorithmStochasticPermanence = False
learningAlgorithmHebbian = True	#strengthen weights of successfully activated neurons

learningAlgorithmHebbianFinalLayer = True	#only apply backprop (effective hebbian) learning at final layer

#supportSkipLayers = True #fully connected skip layer network	#TODO: add support for skip layers	#see ANNtf2_algorithmFBANN for template

debugSingleLayerNetwork = False
debugFastTrain = False

#forward excitatory connections;
W = {}
B = {}
if(learningAlgorithmStochastic):
	Wbackup = {}
	Bbackup = {}
useBinaryWeights = False

Wmean = 0.0
WstdDev = 0.05
	
if(learningAlgorithmHebbianFinalLayer):
	positiveExcitatoryWeightsFinalLayer = False	#allow negative weights on final layer to emulate standard backprop/hebbian learning
	
if(learningAlgorithmStochastic):
	inhibitionFactor1 = 1.0
	inhibitionFactor2 = -(WstdDev*2)		#inhibition contributes a significant (nullifying) effect on layer activation	#CHECKTHIS: requires calibration
	positiveExcitatoryWeights = True	#mandatory
	enableInhibitionTrainSpecificLayerOnly = True
	applyInhibitoryNetworkDuringTest = False
	randomlyActivateWeightsDuringTrain = False
	#learning objective functions:
		#1: maximise the signal (ie successfully uninhibited) across multiple batches (entire dataset)
		#2: ensure that all layer neurons receive even activation across multiple batches (entire dataset)
	metric1Weighting = 1.0
	metric2Weighting = 1000.0	#normalise metric2Weighting relative to metric1Weighting; eg metric1 =  0.9575, metric2 =  0.000863842
	numberStochasticIterations = 10
	updateParameterSubsetSimultaneously = False	#current tests indiciate this is not required/beneficial with significantly high batch size
	if(updateParameterSubsetSimultaneously):
		numberOfSubsetsTrialledPerBaseParameter = 10	#decreases speed, but provides more robust parameter updates
		parameterUpdateSubsetSize = 5	#decreases speed, but provides more robust parameter updates
	else:
		numberOfSubsetsTrialledPerBaseParameter = 1
		parameterUpdateSubsetSize = 1	
	NETWORK_PARAM_INDEX_TYPE = 0
	NETWORK_PARAM_INDEX_LAYER = 1
	NETWORK_PARAM_INDEX_H_CURRENT_LAYER = 2
	NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER = 3
	NETWORK_PARAM_INDEX_VARIATION_DIRECTION = 4
	learningRate = 0.0	#defined by defineTrainingParametersLIANN
elif(learningAlgorithmStochasticPermanence):
	inhibitionFactor1 = 1.0
	inhibitionFactor2 = -(WstdDev*2)	#-1.0		#inhibition contributes a significant (nullifying) effect on layer activation	#CHECKTHIS: requires calibration
	positiveExcitatoryWeights = True	#only allow positive excitatory neuron weights
	enableInhibitionTrainSpecificLayerOnly = False	#CHECKTHIS (set True)
	applyInhibitoryNetworkDuringTest = True	#CHECKTHIS (set False)
	randomlyActivateWeightsDuringTrain = False
	Wpermanence = {}
	Bpermanence = {}
	WpermanenceInitial = 0.1
	BpermanenceInitial = 0.1
	WpermanenceUpdateRate = 0.1
	BpermanenceUpdateRate = 0.1
	permanenceNumberBatches = 10	#if permanenceUpdateRate=1, average number of batches to reset W to random values
	solidificationRate = 0.1
elif(learningAlgorithmHebbian):
	inhibitionFactor1 = 1.0
	inhibitionFactor2 = -(WstdDev*4)	#inhibition contributes a significant (nullifying) effect on layer activation	#CHECKTHIS: requires calibration
	positiveExcitatoryWeights = True	#mandatory
	enableInhibitionTrainSpecificLayerOnly = True	#CHECKTHIS
	applyInhibitoryNetworkDuringTest = False
	randomlyActivateWeightsDuringTrain = True	#randomly activate x weights (simulating input at simulataneous time interval t)
	if(randomlyActivateWeightsDuringTrain):
		randomlyActivateWeightsProbability = 1.0
	WinitialisationFactor = 1.0	#initialise network with relatively low weights	#network will be trained (weights will be increased) up until point where activation inhibited
	BinitialisationFactor = 1.0	#NOTUSED
	learningRate = 0.0	#defined by defineTrainingParametersLIANN
	weightDecay = True	#constant neural net weight decay, such that network can be continuously trained
	if(weightDecay):
		weightDecayRate = 0.0	#defined by defineTrainingParametersLIANN		
	maxWeightUpdateThreshold = True	#max threshold weight updates to learningRate	
	#TODO: ensure learning algorithm does not result in runnaway weight increases
	if(randomlyActivateWeightsDuringTrain):
		inhibitionFactor1 = inhibitionFactor1
		inhibitionFactor2 = (inhibitionFactor2*randomlyActivateWeightsProbability)	#the lower the average activation, the lower the inhibition

	
#lateral inhibitory connections (incoming/outgoing);
IWi = {}
IBi = {}
IWo = {}
IBiWeights = inhibitionFactor1	#WstdDev*2	#0.5	#0.3	#need at least 1/IBiWeights active neurons per layer for the inhibitory neuron to become activated	#CHECKTHIS: requires calibration
IBoWeights = inhibitionFactor2

#Network parameters
n_h = []
In_h = []
numberOfLayers = 0
numberOfNetworks = 0


#note high batchSize is required for learningAlgorithmStochastic algorithm objective functions (>= 100)
def defineTrainingParametersLIANN(dataset):
	global learningRate
	global weightDecayRate	
	
	if(learningAlgorithmStochastic):
		learningRate = 0.001
	if(learningAlgorithmHebbian):
		learningRate = 0.001
		weightDecayRate = learningRate/10.0	#CHECKTHIS	#will depend on learningRate
	batchSize = 100	#3	#100
	numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000
		
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParametersLIANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	global In_h
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet, generateLargeNetwork=False)
	
	In_h = copy.copy(n_h)	#create one inhibitory neuron for every excitatory neuron
	
	return numberOfLayers
	

def defineNeuralNetworkParametersLIANN():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal(mean=Wmean, stddev=WstdDev)
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l in range(1, numberOfLayers+1):
			#forward excitatory connections;
			EWlayer = randomNormal([n_h[l-1], n_h[l]]) 
			EBlayer = tf.zeros(n_h[l])
			if(learningAlgorithmHebbian):
				EWlayer = tf.multiply(EWlayer, WinitialisationFactor)
				EBlayer = tf.multiply(EBlayer, BinitialisationFactor)
			if(positiveExcitatoryWeights):
				if((l < numberOfLayers) or positiveExcitatoryWeightsFinalLayer):
					EWlayer = tf.abs(EWlayer)
			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(EWlayer)
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(EBlayer)
			
			if(learningAlgorithmStochastic):
				Wbackup[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(W[generateParameterNameNetwork(networkIndex, l, "W")])
				Bbackup[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(B[generateParameterNameNetwork(networkIndex, l, "B")])			
			elif(learningAlgorithmStochasticPermanence):
				EWlayerPermanence = tf.multiply(tf.ones([n_h[l-1], n_h[l]]), WpermanenceInitial)
				EBlayerPermanence = tf.multiply(tf.ones(n_h[l]), BpermanenceInitial)
				Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")] = tf.Variable(EWlayerPermanence)
				Bpermanence[generateParameterNameNetwork(networkIndex, l, "Bpermanence")] = tf.Variable(EBlayerPermanence)
			
						
			#lateral inhibitory connections (incoming/outgoing);
			#do not currently train inhibitory weights;
			IWilayer = tf.multiply(tf.ones([n_h[l], In_h[l]]), IBiWeights)		#CHECKTHIS: inhibitory neuron firing is a function of current (lateral) layer (not previous layer)
			IBilayer = tf.zeros(In_h[l])
			IWolayer = tf.multiply(tf.ones([In_h[l], n_h[l]]), IBoWeights)
			IWi[generateParameterNameNetwork(networkIndex, l, "IWi")] = tf.Variable(IWilayer)
			IBi[generateParameterNameNetwork(networkIndex, l, "IBi")] = tf.Variable(IBilayer)
			IWo[generateParameterNameNetwork(networkIndex, l, "IWo")] = tf.Variable(IWolayer)
			

def neuralNetworkPropagationLIANNtest(x, networkIndex=1):
	return neuralNetworkPropagationLIANN(x, None, networkIndex, trainWeights=False)

def neuralNetworkPropagationLIANNtrain(x, y, networkIndex=1):
	if(enableInhibitionTrainSpecificLayerOnly):
		for l in range(1, numberOfLayers+1):
			if(l < numberOfLayers):
				return neuralNetworkPropagationLIANN(x, None, networkIndex, trainWeights=True, layerToTrain=l)
	else:
		return neuralNetworkPropagationLIANN(x, None, networkIndex, trainWeights=True, layerToTrain=None)	
	
		
	
def neuralNetworkPropagationLIANN(x, y, networkIndex=1, trainWeights=False, layerToTrain=None):

	randomNormal = tf.initializers.RandomNormal()
	
	if(trainWeights):
		if(enableInhibitionTrainSpecificLayerOnly):
			maxLayer = layerToTrain
		else:
			maxLayer = numberOfLayers
	else:
		maxLayer = numberOfLayers

	if(useBinaryWeights):
		variationDirections = 1
	else:
		variationDirections = 2
			
	AprevLayer = x
	for l in range(1, maxLayer+1):
					
		trainLayer = False
		enableInhibition = False
		randomlyActivateWeights = False
		if(trainWeights):
			if(enableInhibitionTrainSpecificLayerOnly):
				if(l == layerToTrain):
					#enableInhibition = False
					enableInhibition = True
					trainLayer = True
			else:
				if(l < numberOfLayers):
					enableInhibition = True
					trainLayer = True
			if(randomlyActivateWeightsDuringTrain):
				randomlyActivateWeights = True
		else:
			if(applyInhibitoryNetworkDuringTest):
				enableInhibition = True
	
		finalLayerHebbian = False	
		if(learningAlgorithmHebbianFinalLayer):
			if(l == numberOfLayers):
				finalLayerHebbian = True			
		if(finalLayerHebbian):
			enableInhibition = False
				
		
		if(finalLayerHebbian):
			A, Z, _ = forwardIteration(networkIndex, AprevLayer, l, enableInhibition, randomlyActivateWeights)
		else:			
			if(trainLayer):
				#CHECKTHIS: verify learning algorithm (how to modify weights to maximise independence between neurons on each layer)

				if(learningAlgorithmStochastic):
					#code from ANNtf2_algorithmLREANN_expSUANN;
					for s in range(numberStochasticIterations):
						for hIndexCurrentLayer in range(0, n_h[l]):
							for hIndexPreviousLayer in range(0, n_h[l-1]+1):
								if(hIndexPreviousLayer == n_h[l-1]):	#ensure that B parameter updates occur/tested less frequently than W parameter updates
									parameterTypeWorB = 0
								else:
									parameterTypeWorB = 1
								for variationDirectionInt in range(variationDirections):

									networkParameterIndexBase = (parameterTypeWorB, l, hIndexCurrentLayer, hIndexPreviousLayer, variationDirectionInt)
									
									Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, l, enableInhibition, randomlyActivateWeights)
									metricBase = calculateMetric(Afinal)
									
									for subsetTrialIndex in range(0, numberOfSubsetsTrialledPerBaseParameter):

										accuracyImprovementDetected = False

										currentSubsetOfParameters = []
										currentSubsetOfParameters.append(networkParameterIndexBase)

										for s in range(1, parameterUpdateSubsetSize):
											networkParameterIndex = getRandomNetworkParameter(networkIndex, currentSubsetOfParameters)
											currentSubsetOfParameters.append(networkParameterIndex)

										for s in range(0, parameterUpdateSubsetSize):
											networkParameterIndex = currentSubsetOfParameters[s]

											if(not useBinaryWeights):
												if(networkParameterIndex[NETWORK_PARAM_INDEX_VARIATION_DIRECTION] == 1):
													variationDiff = learningRate
												else:
													variationDiff = -learningRate		
							
											if(networkParameterIndex[NETWORK_PARAM_INDEX_TYPE] == 1):
												#Wnp = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")].numpy()
												#currentVal = Wnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
												currentVal = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

												#print("currentVal = ", currentVal)
												#print("W1 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
												if(useBinaryWeights):
													if(useBinaryWeightsReduceMemoryWithBool):
														newVal = not currentVal
													else:
														newVal = float(not bool(currentVal))
														#print("newVal = ", newVal)
												else:
													newVal = currentVal + variationDiff
																								
												W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)

												#print("W2 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
											else:
												#Bnp = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")].numpy()
												#currentVal = Bnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
												currentVal = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

												if(useBinaryWeights):
													if(useBinaryWeightsReduceMemoryWithBool):
														newVal = not currentVal
													else:
														newVal = float(not bool(currentVal))
												else:
													newVal = currentVal + variationDiff
												B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)

										Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, l, enableInhibition, randomlyActivateWeights)
										metricAfterStochasticUpdate = calculateMetric(Afinal)
										#print("metricBase = ", metricBase)
										#print("metricAfterStochasticUpdate = ", metricAfterStochasticUpdate)
						
										if(metricAfterStochasticUpdate > metricBase):
											#print("(metricAfterStochasticUpdate > metricBase)")
											accuracyImprovementDetected = True
											metricBase = metricAfterStochasticUpdate
										#else:
											#print("(metricAfterStochasticUpdate < metricBase)")

										if(accuracyImprovementDetected):
											#retain weight update
											Wbackup[generateParameterNameNetwork(networkIndex, l, "W")].assign(W[generateParameterNameNetwork(networkIndex, l, "W")])
											Bbackup[generateParameterNameNetwork(networkIndex, l, "B")].assign(B[generateParameterNameNetwork(networkIndex, l, "B")])	
										else:
											#restore weights
											W[generateParameterNameNetwork(networkIndex, l, "W")].assign(Wbackup[generateParameterNameNetwork(networkIndex, l, "W")])
											B[generateParameterNameNetwork(networkIndex, l, "B")].assign(Bbackup[generateParameterNameNetwork(networkIndex, l, "B")])								
				elif(learningAlgorithmStochasticPermanence):
					Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, l, enableInhibition, randomlyActivateWeights)

					#update W/B permanence;
					Afinal2D = tf.reduce_mean(Afinal, axis=0)	#average across batch
					Afinal2D = tf.expand_dims(Afinal2D, axis=0)	#make compatible shape to W
					WpermanenceUpdate = tf.multiply(Afinal2D, WpermanenceUpdateRate)	#verify that broadcasting works
					WpermanenceNew = tf.add(Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")], WpermanenceUpdate)	#increase the permanence of neuron weights that successfully fired
					Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")] = WpermanenceNew
					print("WpermanenceUpdate = ", WpermanenceUpdate)

					#stochastically modify weights based on permanence values:
					Wupdate = randomNormal([n_h[l-1], n_h[l]])
					Wupdate = tf.divide(Wupdate, Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")])
					Wupdate = tf.divide(Wupdate, permanenceNumberBatches)
					Wnew = tf.add(W[generateParameterNameNetwork(networkIndex, l, "W")], Wupdate)
					W[generateParameterNameNetwork(networkIndex, l, "W")] = Wnew
					#print("Wupdate = ", Wupdate)
				elif(learningAlgorithmHebbian):
					AW = W[generateParameterNameNetwork(networkIndex, l, "W")] 
					Afinal, Zfinal, EWactive = forwardIteration(networkIndex, AprevLayer, l, enableInhibition, randomlyActivateWeights)
					print("Afinal = ", Afinal)
					AWcontribution = tf.matmul(tf.transpose(AprevLayer), Afinal)	#increase excitatory weights that contributed to the output signal	#hebbian
					if(maxWeightUpdateThreshold):
						AWcontribution = tf.minimum(AWcontribution, 1.0)
					if(randomlyActivateWeights):
						#do not apply weight updates to temporarily suppressed weights [CHECKTHIS];
						AWcontribution = tf.multiply(AWcontribution, EWactive)			
					AWupdate = tf.multiply(AWcontribution, learningRate)
					#print("AWupdate = ", AWupdate)
					AW = tf.add(AW, AWupdate)
					if(weightDecay):
						#apply decay to all weights;
						AWdecay = -weightDecayRate
						#print("AWdecay = ", AWdecay)
						AW = tf.add(AW, AWdecay)
						#do not allow weights fall below zero [CHECKTHIS];
						AW = tf.maximum(AW, 0)
						#print("AWdecay = ", AWdecay)
					W[generateParameterNameNetwork(networkIndex, l, "W")] = AW

				A, Z, _ = forwardIteration(networkIndex, AprevLayer, l, enableInhibition=False, randomlyActivateWeights=False)	#in case !learningAlgorithmHebbianFinalLayer
			else:
				A, Z, _ = forwardIteration(networkIndex, AprevLayer, l, enableInhibition, randomlyActivateWeights)

			if(learningAlgorithmHebbianFinalLayer):
				A = tf.stop_gradient(A)
				
			AprevLayer = A

	return tf.nn.softmax(Z)
	
def forwardIteration(networkIndex, AprevLayer, l, enableInhibition=False, randomlyActivateWeights=False):

	#forward excitatory connections;
	EWactive = None
	EW = W[generateParameterNameNetwork(networkIndex, l, "W")]
	if(randomlyActivateWeights):
		#print("EW = ", EW)
		EWactive = tf.less(tf.random.uniform(shape=EW.shape), randomlyActivateWeightsProbability)	#initialised from 0.0 to 1.0
		EWactive = tf.dtypes.cast(EWactive, dtype=tf.dtypes.float32) 
		#print("EWactive = ", EWactive)
		#EWactive = tf.dtypes.cast(tf.random.uniform(shape=EW.shape, minval=0, maxval=2, dtype=tf.dtypes.int32), dtype=tf.dtypes.float32)
		EW = tf.multiply(EW, EWactive)
	Z = tf.add(tf.matmul(AprevLayer, EW), B[generateParameterNameNetwork(networkIndex, l, "B")])
	A = activationFunction(Z)

	#lateral inhibitory connections (incoming/outgoing);
	if(enableInhibition):
		#if((l < numberOfLayers) or positiveExcitatoryWeightsFinalLayer):

		#print("AprevLayer = ", AprevLayer)
		#print("Z = ", Z)

		IZi = tf.matmul(A, IWi[generateParameterNameNetwork(networkIndex, l, "IWi")])	#CHECKTHIS: inhibitory neuron firing is a function of current (lateral) layer (not previous layer)
		IAi = activationFunction(IZi)
		#print("IZi = ", IZi)
		#print("IAi = ", IAi)
		IZo = tf.matmul(IAi, IWo[generateParameterNameNetwork(networkIndex, l, "IWo")])
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		#print("IZo = ", IZo)
		
		#final activations;
		Zfinal = tf.add(Z, IZo)
		#print("Zfinal = ", Zfinal)
		Afinal = activationFunction(Zfinal)
	else:
		Zfinal = Z
		Afinal = A
	
	return Afinal, Zfinal, EWactive
				
def calculateMetric(Afinal):
	#learning objective functions:
	#1: maximise the signal (ie successfully uninhibited) across multiple batches (entire dataset)
	#2: ensure that all layer neurons receive even activation across multiple batches (entire dataset)		
	
	#print("Afinal = ", Afinal) 

	AfinalThresholded = tf.greater(Afinal, 0.0)	#threshold signal such that higher average weights are not preferenced
	AfinalThresholded = tf.dtypes.cast(AfinalThresholded, dtype=tf.dtypes.float32)	
	#print("Afinal = ", Afinal)
	#print("AfinalThresholded = ", AfinalThresholded)
	
	metric1 = tf.reduce_mean(AfinalThresholded)	#average output across batch, across layer
	
	#stdDevAcrossLayer = tf.math.reduce_std(Afinal, axis=1)	#stddev calculated across layer [1 result per batch index]
	#metric2 = tf.reduce_mean(stdDevAcrossLayer)	#average output across batch
	
	stdDevAcrossBatches = tf.math.reduce_mean(Afinal, axis=0)	 #for each dimension (k neuron in layer); calculate the mean across all batch indices
	metric2 = tf.math.reduce_std(stdDevAcrossBatches)	#then calculate the std dev across these values
	
	metric1 = metric1.numpy()
	metric2 = metric2.numpy()
	#print("metric1 = ", metric1)
	#print("metric2 = ", metric2)
				
	metric1 = metric1*metric1Weighting
	metric2 = metric2*metric2Weighting
	#print("metric1 = ", metric1)
	#print("metric2 = ", metric2)
	if(metric2 != 0):
		metric = metric1/metric2
	else:
		metric = 0.0
	
	return metric

def getRandomNetworkParameter(networkIndex, currentSubsetOfParameters):
	
	foundNewParameter = False
	while not foundNewParameter:
	
		variationDirection = random.randint(2)
		layer = random.randint(1, len(n_h))
		parameterTypeWorBtemp = random.randint(n_h[layer-1]+1)	#ensure that B parameter updates occur/tested less frequently than W parameter updates	#OLD: random.randint(2)	
		if(parameterTypeWorBtemp == n_h[layer-1]):
			parameterTypeWorB = 0
		else:
			parameterTypeWorB = 1
		hIndexCurrentLayer = random.randint(n_h[layer])	#randomNormal(n_h[l])
		hIndexPreviousLayer = random.randint(n_h[layer-1]) #randomNormal(n_h[l-1])
		networkParameterIndex = (parameterTypeWorB, layer, hIndexCurrentLayer, hIndexPreviousLayer, variationDirection)
	
		matches = [item for item in currentSubsetOfParameters if item == networkParameterIndex]
		if len(matches) == 0:
			foundNewParameter = True
			
	return networkParameterIndex				


def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A
	

