"""SANItf2_algorithmSANIglobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SANItf2.py

# Usage:
see SANItf2.py

# Description:
SANItf algorithm SANI (Sequentially Activated Neuronal Input) global definitions

"""

import tensorflow as tf
import numpy as np
import ANNtf2_globalDefs

#select algorithmSANI:
#algorithmSANI = "sharedModulesNonContiguousFullConnectivity"
#algorithmSANI = "sharedModulesBinary"
algorithmSANI = "sharedModules"
#algorithmSANI = "repeatedModules"

createSmallNetworkForDebug = True

useSequentialInputs = True
if(useSequentialInputs):
	numberOfSequentialInputs = 2	#2	#3	#1 - no sequential input requirement enforced
else:
	numberOfSequentialInputs = 1

useTcontiguity = False

useFullConnectivitySparsity = False
useLearningRuleBackpropagation = False
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	useLearningRuleBackpropagation = True	#optional (untested)
	useFullConnectivitySparsity = True	#sparsity is defined within fully connected weights
	supportFullConnectivity = True	#full connectivity between layers	
	supportFeedback = False	#optional 
	useHebbianLearningRule = False
#	if(useHebbianLearningRule):
#		useFullConnectivitySparsity = True
#		useHebbianLearningRulePositiveWeights = True
#		neuronActivationFiringThreshold = 1.0
#		useHebbianLearningRuleApply = True
#		if(useHebbianLearningRuleApply):
#			#not currently compatible with supportFullConnectivity:supportFeedback
#			#these parameters require calibration:
#			hebbianLearningRate = 0.01
#			minimumConnectionWeight = 0.0
#			maximumConnectionWeight = 1.0
elif(algorithmSANI == "sharedModulesBinary"):
	useTcontiguity = False	#optional (default:False from 19 April 2022)
	useLearningRuleBackpropagation = True	#optional (untested)
	supportFullConnectivity = False	#unimplemented (could be added in future)
	supportFeedback = False	#unimplemented
elif(algorithmSANI == "sharedModules"):
	useTcontiguity = False	#optional (default:False from 19 April 2022)
	useLearningRuleBackpropagation = True	#optional (untested)
	supportFullConnectivity = False	#unimplemented (could be added in future)
	supportFeedback = False	#unimplemented
elif(algorithmSANI == "repeatedModules"):
	useTcontiguity = False	#mandatory	#CHECKTHIS
	useLearningRuleBackpropagation = True	#optional (untested)
	supportFullConnectivity = False	#unimplemented (could be added in future)
	supportFeedback = False	#unimplemented
#supportFeedback note: activation/A must be maintained across multiple iteration forward propagation through layers
	#currently requires SANIsharedModules=True
	#SANIsharedModules=False would need to be upgraded to perform multiple forward pass iterations
	
veryLargeInt = 99999999


if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	SANIsharedModules = True	#optional
elif(algorithmSANI == "sharedModulesBinary"):
	SANIsharedModules = True	#mandatory (only coded implementation)
elif(algorithmSANI == "sharedModules"):
	SANIsharedModules = True	#mandatory	(only coded implementation)
elif(algorithmSANI == "repeatedModules"): 	
	SANIsharedModules = False	#mandatory (only coded implementation)
#SANIsharedModules note: uses shifting input x feed, enabling identical input subsets (eg phrases/subreferencesets) irrespective of their sentence position to be sent to same modules/neurons


if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	allowMultipleSubinputsPerSequentialInput = True	#implied variable (granted via full connectivity)
	inputNumberFeaturesForCurrentWordOnly = True	#optional
elif(algorithmSANI == "sharedModulesBinary"):
	allowMultipleSubinputsPerSequentialInput = True	#required #originally set as False
	inputNumberFeaturesForCurrentWordOnly = True	#mandatory (only coded implementation)

	resetSequentialInputs = True
	if(useTcontiguity):
		resetSequentialInputsIfOnlyFirstInputValid = True	#see GIA_TXT_REL_TRANSLATOR_NEURAL_NETWORK_SEQUENCE_GRAMMAR development history for meaning and algorithmic implications of this feature
	if(resetSequentialInputs):
		doNotResetNeuronOutputUntilAllSequentialInputsActivated = True
		
	useSparseTensors = True	#mandatory
elif(algorithmSANI == "sharedModules"):
	allowMultipleSubinputsPerSequentialInput = True
	
	allowMultipleContributingSubinputsPerSequentialInput = False	#initialise
	if(allowMultipleSubinputsPerSequentialInput):
		if(useTcontiguity):
			allowMultipleContributingSubinputsPerSequentialInput = False	#optional	#whether only 1 subinput can be fired to activate a sequential input	
			if(allowMultipleContributingSubinputsPerSequentialInput):
				inputNumberFeaturesForCurrentWordOnly = False	#the convolutional window (kernel) captures x words every time is slided to the right
			else:
				inputNumberFeaturesForCurrentWordOnly = True
		else:
			inputNumberFeaturesForCurrentWordOnly = True
	else:
		inputNumberFeaturesForCurrentWordOnly = True

	resetSequentialInputs = True
	if(useTcontiguity):
		resetSequentialInputsIfOnlyFirstInputValid = True	#see GIA_TXT_REL_TRANSLATOR_NEURAL_NETWORK_SEQUENCE_GRAMMAR development history for meaning and algorithmic implications of this feature
		if(resetSequentialInputsIfOnlyFirstInputValid):
			if(allowMultipleContributingSubinputsPerSequentialInput):
				averageTimeChangeOfNewInputRequiredForReset = 1
	if(resetSequentialInputs):
		doNotResetNeuronOutputUntilAllSequentialInputsActivated = True

	if(allowMultipleSubinputsPerSequentialInput):
		if(allowMultipleContributingSubinputsPerSequentialInput):
			useSparseTensors = False	#optional
		else:
			useSparseTensors = True		#mandatory	#FUTURE: upgrade code to remove this requirement 
	else:
		useSparseTensors = True	#mandatory	#sparse tensors are used
elif(algorithmSANI == "repeatedModules"): 	
	allowMultipleSubinputsPerSequentialInput = False
	useSparseTensors = True	#mandatory
	inputNumberFeaturesForCurrentWordOnly = False	#NA (not used)

if(allowMultipleSubinputsPerSequentialInput):
	layerSizeConvergence = True	#CHECKTHIS
else:
	layerSizeConvergence = False

#set parameters oneSequentialInputHasOnlyOneSubinput:
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	oneSequentialInputHasOnlyOneSubinput = False	
elif(algorithmSANI == "sharedModulesBinary"):		
	expectNetworkConvergence = False
	if(expectNetworkConvergence):
		#if(numberOfSequentialInputs == 2):
		oneSequentialInputHasOnlyOneSubinput = True	#conditional probability determination of events
	else:
		oneSequentialInputHasOnlyOneSubinput = False
elif(algorithmSANI == "sharedModules"):
	if(useSparseTensors):	#FUTURE: upgrade code to remove this requirement
		if(allowMultipleSubinputsPerSequentialInput):
			#if(numberOfSequentialInputs == 2):
			oneSequentialInputHasOnlyOneSubinput = True	#conditional probability determination of events
		else:
			oneSequentialInputHasOnlyOneSubinput = False
	else:
		oneSequentialInputHasOnlyOneSubinput = False
elif(algorithmSANI == "repeatedModules"):
	oneSequentialInputHasOnlyOneSubinput = False
if(oneSequentialInputHasOnlyOneSubinput):
	firstSequentialInputHasOnlyOneSubinput = True #use combination of allowMultipleSubinputsPerSequentialInput for different sequential inputs;  #1[#2] sequential input should allow multiple subinputs, #2[#1] sequential input should allow single subinput
	if(firstSequentialInputHasOnlyOneSubinput):
		lastSequentialInputHasOnlyOneSubinput = False
	else:
		lastSequentialInputHasOnlyOneSubinput = True

#set parameters enforceTcontiguityConstraints:
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	enforceTcontiguityConstraints = False
elif(algorithmSANI == "sharedModulesBinary"):		
	if(useTcontiguity):
		enforceTcontiguityConstraints = True
	else:
		enforceTcontiguityConstraints = False
elif(algorithmSANI == "sharedModules"):	
	if(useTcontiguity):
		if(allowMultipleSubinputsPerSequentialInput):
			if(useSparseTensors):
				if(not allowMultipleContributingSubinputsPerSequentialInput):
					if(useTcontiguity):
						enforceTcontiguityConstraints = True
					else:
						enforceTcontiguityConstraints = False
				else:
					enforceTcontiguityConstraints = False
			else:
				enforceTcontiguityConstraints = False
		else:
			enforceTcontiguityConstraints = True
	else:
		enforceTcontiguityConstraints = False
elif(algorithmSANI == "repeatedModules"): 
	if(useTcontiguity):
		enforceTcontiguityConstraints = True
	else:
		enforceTcontiguityConstraints = False
if(enforceTcontiguityConstraints):
	enforceTcontiguityBetweenSequentialInputs = True
	enforceTcontiguityTakeEncapsulatedSubinputWithMinTvalue = True	#method to decide between subinput selection/parse tree generation
	enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW = True
	if(not useLearningRuleBackpropagation):
		enforceTcontiguityStartAndEndOfSequence = True
	

if(inputNumberFeaturesForCurrentWordOnly):
	numberOfWordsInConvolutionalWindowSeen = 1	#always 1
else:
	numberOfWordsInConvolutionalWindowSeen = 10
	
#set parameters performSummationOfSubInputsWeighted/useLastSequentialInputOnly/numberOfWordsInConvolutionalWindowSeen:
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	performThresholdOfSubInputsNonlinear = True	#default: apply non-linear activation function to sequential input (use Aseq)
	if(not performThresholdOfSubInputsNonlinear):
		performThresholdOfSubInputsBinary = True		#optional	#simple thresholding function (activations are normalised to 1.0)
		if(not performThresholdOfSubInputsBinary):
			sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)	#should this be used with backprop?
	sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)	#should this be used with backprop?
		
	performSummationOfSubInputs = True	#mandatory (implied)
	if(performSummationOfSubInputs):
		performSummationOfSubInputsWeighted = True	#mandatory (implied)
		if(performThresholdOfSubInputsNonlinear):
			performSummationOfSubInputsNonlinear = True		#default: apply non-linear activation function to sequential input (use Aseq)
		else:
			performSummationOfSubInputsNonlinear = False
	
	performSummationOfSequentialInputs = True	#optional (else just take the last Zseq/Aseq values)
	if(performSummationOfSequentialInputs):
		performSummationOfSequentialInputsWeighted = False	#optional: multiply sequential input activations (Zseq/Aseq) by a matrix - otherwise just sum/average them together (or take a pass condition)
		if(performSummationOfSequentialInputsWeighted):
			sequentialInputCombinationModeSummationAveraged = False	#mandatory
			if(performSummationOfSubInputsNonlinear):
				performSummationOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Aseq multiplied by a matrix
			else:
				performSummationOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Zseq multiplied by a matrix
		else:
			sequentialInputCombinationModeSummationAveraged = True	#default: average values across seq (else sum)
			if(performSummationOfSubInputsNonlinear):
				performSummationOfSequentialInputsNonlinear = False	#default: !apply non-linear activation function neuron: use A value from typically all Aseq summed/averaged			
			else:
				performSummationOfSequentialInputsNonlinear = True 	#default: apply non-linear activation function neuron: use Z value from typically all Zseq summed/averaged
	else:
		pass	#if(performSummationOfSubInputsBinary): simple thresholding function (activations are normalised to 1.0)
elif(algorithmSANI == "sharedModulesBinary"):		
	performSummationOfSubInputsWeighted = False	#mandatory
	useLastSequentialInputOnly = True	#implied variable (not used)
	performSummationOfSequentialInputsWeighted = False	#mandatory (implied)
elif(algorithmSANI == "sharedModules"):
	performThresholdOfSubInputsNonlinear = True	#default: apply non-linear activation function to sequential input (use Aseq)
	if(not performThresholdOfSubInputsNonlinear):
		sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)	#should this be used with backprop?
		
	if(allowMultipleSubinputsPerSequentialInput):
		if(allowMultipleContributingSubinputsPerSequentialInput):
			#[multiple contributing subinputs per sequential input] #each sequential input can detect a pattern of activation from the previous layer
			performSummationOfSubInputs = True	#mandatory (implied)
			performSummationOfSubInputsWeighted = True	#mandatory?
			if(performThresholdOfSubInputsNonlinear):
				performSummationOfSubInputsNonlinear = True
			else:
				performSummationOfSubInputsNonlinear = False
		else:
			performSummationOfSubInputs = False	#optional though by algorithm design: False
			performSummationOfSubInputsWeighted = False	#will take (True: most weighted) (False: any) active time contiguous subinput
			if(performThresholdOfSubInputsNonlinear):
				performSummationOfSubInputsNonlinear = True
			else:
				performSummationOfSubInputsNonlinear = False

		performSummationOfSequentialInputs = True	#optional (else just take the last Zseq/Aseq values)
		if(performSummationOfSequentialInputs):
			performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
			performSummationOfSequentialInputsWeighted = False	#optional: multiply sequential input activations (Zseq/Aseq) by a matrix - otherwise just sum/average them together (or take a pass condition)
			if(performSummationOfSequentialInputsWeighted):
				sequentialInputCombinationModeSummationAveraged = False	#mandatory
				if(performSummationOfSubInputsNonlinear):
					performSummationOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Aseq multiplied by a matrix
				else:
					performSummationOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Zseq multiplied by a matrix
			else:
				sequentialInputCombinationModeSummationAveraged = True	#default: average values across seq (else sum)
				if(performSummationOfSubInputsNonlinear):
					performSummationOfSequentialInputsNonlinear = False	#default: !apply non-linear activation function neuron: use A value from typically all Aseq summed/averaged			
				else:
					performSummationOfSequentialInputsNonlinear = True 	#default: apply non-linear activation function neuron: use Z value from typically all Zseq summed/averaged
		else:
			useLastSequentialInputOnly = True	#implied variable (not used)

	else:
		performSummationOfSubInputsWeighted = False

		performSummationOfSequentialInputs = True
		if(performSummationOfSequentialInputs):
			performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
			performSummationOfSequentialInputsWeighted = True	#optional: multiply sequential input activations (Zseq/Aseq) by a matrix - otherwise just sum/average them together (or take a pass condition)
			if(performSummationOfSequentialInputsWeighted):
				sequentialInputCombinationModeSummationAveraged = False	#mandatory
				performSummationOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Zseq multiplied by a matrix
			else:
				sequentialInputCombinationModeSummationAveraged = True	#default: average values across seq (else sum)
				performSummationOfSequentialInputsNonlinear = True 	#default: apply non-linear activation function neuron: use Z value from typically all Zseq summed/averaged
		else:
			useLastSequentialInputOnly = True	#implied variable (not used)
elif(algorithmSANI == "repeatedModules"): 
	performThresholdOfSubInputsNonlinear = True	#default: apply non-linear activation function to sequential input (use Aseq)
	if(not performThresholdOfSubInputsNonlinear):
		sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)	#should this be used with backprop?
		
	performSummationOfSequentialInputsWeighted = False	#initialise
	if(allowMultipleSubinputsPerSequentialInput):
		#[multiple subinputs per sequential input] #each sequential input can detect a pattern of activation from the previous layer

		performIndependentSubInputValidation = True
		performSummationOfSubInputs = True	#else take sub input with max input signal*weight
		if(performSummationOfSubInputs):
			performSummationOfSubInputsWeighted = True	#determines if backprop is required to update weight matrix associated with inputs to a sequential input?
			if(performThresholdOfSubInputsNonlinear):
				performSummationOfSubInputsNonlinear = True
			else:
				performSummationOfSubInputsNonlinear = False
		else:
			performSummationOfSubInputsWeighted = False
			if(performThresholdOfSubInputsNonlinear):
				performSummationOfSubInputsNonlinear = True
			else:
				performSummationOfSubInputsNonlinear = False	

		performSummationOfSequentialInputs = True	#optional (else just take the last Zseq/Aseq values)
		if(performSummationOfSequentialInputs):
			performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
			performSummationOfSequentialInputsWeighted = False	#optional: multiply sequential input activations (Zseq/Aseq) by a matrix - otherwise just sum/average them together (or take a pass condition)
			if(performSummationOfSequentialInputsWeighted):
				sequentialInputCombinationModeSummationAveraged = False	#mandatory
				if(performSummationOfSubInputsNonlinear):
					performSummationOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Aseq multiplied by a matrix
				else:
					performSummationOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Zseq multiplied by a matrix
			else:
				sequentialInputCombinationModeSummationAveraged = True	#default: average values across seq (else sum)
				if(performSummationOfSubInputsNonlinear):
					performSummationOfSequentialInputsNonlinear = False	#default: !apply non-linear activation function neuron: use A value from typically all Aseq summed/averaged			
				else:
					performSummationOfSequentialInputsNonlinear = True 	#default: apply non-linear activation function neuron: use Z value from typically all Zseq summed/averaged
		else:
			useLastSequentialInputOnly = True	#implied variable (not used)
		
		numberSubinputsPerSequentialInput = 3 #sparsity

		if(enforceTcontiguityConstraints):
			sequentialityMode = "default"
			#sequentialityMode = "temporalCrossoverAllowed"
			#sequentialityMode = "contiguousInputEnforced"
	else:
		#[single subinput per sequential input] #each sequential input is directly connected to a single neuron on the previous layer

		performIndependentSubInputValidation = False	#always False (ie false by definition because there is only 1 subinput per sequential input)

		performSummationOfSequentialInputs = True
		if(performSummationOfSequentialInputs):
			performSummationOfSequentialInputsWeighted = True	#does backprop require to update weight matrix associated with sequential inputs?
			performSummationOfSequentialInputsNonlinear = True
		else:
			performSummationOfSequentialInputsWeighted = False
			performSummationOfSequentialInputsNonlinear = False

		if(enforceTcontiguityConstraints):
			sequentialityMode = "default"
			#sequentialityMode = "temporalCrossoverAllowed"
			#sequentialityMode = "contiguousInputEnforced"

		
#set parameters record:
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):	
	recordNetworkWeights = False	#need to modify weights not just record them
	if(recordNetworkWeights):
		recordSubInputsWeighted = False	#batchSize must equal 1
		recordSequentialInputsWeighted = False
		recordNeuronsWeighted = False
elif(algorithmSANI == "sharedModulesBinary"):
	if(not ANNtf2_globalDefs.testHarness):	
		recordNetworkWeights = True
		if(recordNetworkWeights):
			recordSubInputsWeighted = True
			recordSequentialInputsWeighted = False	#may not be necessary (only used if can split neuron sequential inputs)
			recordNeuronsWeighted = True
			#FUTURE: prune network neurons/connections based on the relative strength of these weights
	else:
		recordNetworkWeights = False
		recordSubInputsWeighted = False
		recordSequentialInputsWeighted = False
		recordNeuronsWeighted = False
elif(algorithmSANI == "sharedModules"):
	if(useSparseTensors):
		if(allowMultipleSubinputsPerSequentialInput):		
			recordNetworkWeights = True
			if(recordNetworkWeights):
				recordSubInputsWeighted = True
				recordSequentialInputsWeighted = False	#may not be necessary (only used if can split neuron sequential inputs)
				recordNeuronsWeighted = True
				#FUTURE: prune network neurons/connections based on the relative strength of these weights
		else:
			recordNetworkWeights = False	#not yet coded
	else:
		recordNetworkWeights = False	#not yet coded
elif(algorithmSANI == "repeatedModules"): 	
	recordNetworkWeights = False


if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	supportSkipLayers = True
	maxNumberSubinputsPerSequentialInput = -1	#NA
elif(algorithmSANI == "sharedModulesBinary"):
	if(allowMultipleSubinputsPerSequentialInput):	
		if(expectNetworkConvergence):
			maxNumberSubinputsPerSequentialInput = 50	#~approx equal number of prev layer neurons/2 (FUTURE: make dynamic based on layer index)	#number of prior/future events in which to calculate a conditional probability
		else:
			maxNumberSubinputsPerSequentialInput = 1	#sparsity
	supportSkipLayers = True
elif(algorithmSANI == "sharedModules"):
	if(allowMultipleSubinputsPerSequentialInput):
		if(useSparseTensors):
			supportSkipLayers = True
			if(oneSequentialInputHasOnlyOneSubinput):
				maxNumberSubinputsPerSequentialInput = 50	#~approx equal number of prev layer neurons/2	#number of prior/future events in which to calculate a conditional probability
			else:
				maxNumberSubinputsPerSequentialInput = 3	#sparsity
		else:
			supportSkipLayers = True
	else:
		supportSkipLayers = True
else:
	supportSkipLayers = True
	maxNumberSubinputsPerSequentialInput = -1	#NA

	



