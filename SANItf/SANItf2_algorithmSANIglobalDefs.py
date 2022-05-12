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

#select dataset:
#dataset = "POStagSentence"
#dataset = "POStagSequence"
dataset = "wikiXmlDataset"	#high first hidden layer connectivity is required
#SANItf2: dataset content requirements: require at least 2 sentences/sequences/sentences in data file

#performance test/enhancements for development environment only: 
printIO = False
printStatus = False
printGraphVisualisation = False	#tensorflow requires autograph compatible functions for visualisation (dynamic computational graphs defined through eager execution not supported)	#incomplete; requires to convert tensor dynamic initialisations to static initialisations via (unavailable) python preprocessor definitions

debugActivationNormalisation = False	#verify activation/strength across layers
debugTrainSingleBatch = True	#test paramater optimisation by repeatedly training a single batch only
debugPrintGradients = False	#test backpropagation through SANI
debugCountGradients = False
debugTrainLastLayerOnly = False	#train last layer only
debugUseSmallSequenceDataset = True	#increases performance during development	#eg data-POStagSentence-smallBackup
debugUseSmallSentenceLengths = False	#increases performance during development	#eg data-simple-POStagSentence-smallBackup	#Preconditions: requires dataset with at least 2 sentences of size < limitSentenceLengthsSize; for test/train split
debugUseSmallBatchSize = False	#use batchSize=1
debugUseSmallLayers = False	#small number of neurons per layer (not functional)

if(debugActivationNormalisation):
	printStatus = True
	debugUseSmallLayers = True
	debugUseSmallBatchSize = True
	
#parameter configuration (all algorithmSANI):
useLearningRuleBackpropagation = True	#optional (untested)
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	useTcontiguity = False	#mandatory false (not supported)
	useSkipLayers = False	#optional (orig: True)
	useMultipleSubinputsPerSequentialInput = True	#mandatory (implied true)
	activationFunctionThreshold	= 0.0	#default: 0.0	#>=0: reduces probability of neural sequential input firing	#not used
else:
	useTcontiguity = False	#optional (orig: True)
	useSkipLayers = False	#optional (orig: True)
	useMultipleSubinputsPerSequentialInput = True	#optional
	
if((algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):
	useReverseSequentialInputOrder = False	#initialise
	if(useTcontiguity):
		resetSequentialInputs = True	#likely mandatory (development conditions)
		useReverseSequentialInputOrder = True	#likely mandatory (development conditions)
	else:
		resetSequentialInputs = False	#optional	#overwrites first input if valid and resets all sequential inputs
		if(resetSequentialInputs):
			useReverseSequentialInputOrder = True	#optional - required for resetSequentialInputsIfOnlyFirstInputValid	

	overwriteSequentialInputs = False	#initialise		
	if(not resetSequentialInputs):
		overwriteSequentialInputs = True	#mandatory	#overwrites any valid sequential input it encounters
				
if(useMultipleSubinputsPerSequentialInput):
	numberSubinputsPerSequentialInputDefault = 10	#default: 3
useSequentialInputs = True
if(useSequentialInputs):
	numberOfSequentialInputs = 2	#2	#3	#1 - no sequential input requirement enforced
else:
	numberOfSequentialInputs = 1


if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	SANIsharedModules = True	#optional
elif(algorithmSANI == "sharedModulesBinary"):
	SANIsharedModules = True	#mandatory (only coded implementation)
elif(algorithmSANI == "sharedModules"):
	SANIsharedModules = True	#mandatory	(only coded implementation)
elif(algorithmSANI == "repeatedModules"): 	
	SANIsharedModules = False	#mandatory (only coded implementation)
#SANIsharedModules note: uses shifting input x feed, enabling identical input subsets (eg phrases/subreferencesets) irrespective of their sentence position to be sent to same modules/neurons


#POStagSequence dataset contains equal length input sequences, POStagSentence dataset contains arbitarily lengthed input sequences

firstLayerFullConnectivity = False
if(dataset == "wikiXmlDataset"):
	vectorisedInput = True	#word vector input
	vectorisedOutput = True	#word vector output
	if(SANIsharedModules):
		firstLayerFullConnectivity = True	#connect first layer neurons to every feature in input word vector
else:
	vectorisedInput = False
	vectorisedOutput = False	#single/multi class output (one-hot encoded)
	
	
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	supportFullConnectivity = True	#mandatory - full connectivity between layers	
	if(supportFullConnectivity):
		useFullConnectivitySparsity = False	#sparsity is defined by zeroing fully connected weights 
		if(useFullConnectivitySparsity):
			probabilityOfActiveConnection = 0.1	#1-sparsityLevel
		supportFeedback = False	#optional 
	useHebbianLearningRule = False
elif(algorithmSANI == "sharedModulesBinary"):
	supportFullConnectivity = False	#unimplemented (could be added in future)
	supportFeedback = False	#unimplemented
elif(algorithmSANI == "sharedModules"):
	supportFullConnectivity = False	#unimplemented (could be added in future)
	supportFeedback = False	#unimplemented
elif(algorithmSANI == "repeatedModules"):
	supportFullConnectivity = False	#unimplemented (could be added in future)
	supportFeedback = False	#unimplemented

#supportFeedback note: activation/A must be maintained across multiple iteration forward propagation through layers
	#currently requires SANIsharedModules=True
	#SANIsharedModules=False would need to be upgraded to perform multiple forward pass iterations
	
veryLargeInt = 99999999


allowMultipleSubinputsPerSequentialInput = False	#initialise
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	if(useMultipleSubinputsPerSequentialInput):
		#allowMultipleSubinputsPerSequentialInput = True	#implied variable (granted via full connectivity)
		pass
	if(SANIsharedModules):
		inputNumberFeaturesForCurrentWordOnly = True	#optional
	else:
		inputNumberFeaturesForCurrentWordOnly = False	#mandatory
elif(algorithmSANI == "sharedModulesBinary"):
	if(useMultipleSubinputsPerSequentialInput):
		allowMultipleSubinputsPerSequentialInput = True		#originally set as False
	inputNumberFeaturesForCurrentWordOnly = True	#mandatory (only coded implementation)

	resetSequentialInputsIfOnlyFirstInputValid = False	#initialise
	if(resetSequentialInputs):
		if(useReverseSequentialInputOrder):
			resetSequentialInputsIfOnlyFirstInputValid = True	#interpretation: resetSequentialInputsIfOnlyFirstInputValid and no others
		if(useTcontiguity):
			resetSequentialInputsTContiguity = True	#see GIA_TXT_REL_TRANSLATOR_NEURAL_NETWORK_SEQUENCE_GRAMMAR development history for meaning and algorithmic implications of this feature
		doNotResetNeuronOutputUntilAllSequentialInputsActivated = True	#delays disabling neuron output while measuring new sequential inputs
		
	useSparseTensors = True	#mandatory
	
	performTindependentFunctionOfSubInputs = False	#initialise
	if(allowMultipleSubinputsPerSequentialInput):
		if(useTcontiguity):
			performTindependentFunctionOfSubInputs = True	#CHECKTHIS (not yet implemented)
		
elif(algorithmSANI == "sharedModules"):
	if(useMultipleSubinputsPerSequentialInput):
		allowMultipleSubinputsPerSequentialInput = True
	
	allowMultipleContributingSubinputsPerSequentialInput = False	#initialise	#relevant to useTcontiguity only
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

	resetSequentialInputsIfOnlyFirstInputValid = False	#initialise
	if(resetSequentialInputs):
		if(useReverseSequentialInputOrder):
			resetSequentialInputsIfOnlyFirstInputValid = True	#interpretation: resetSequentialInputsIfOnlyFirstInputValid and no others
		if(useTcontiguity):
			resetSequentialInputsTContiguity = True	#see GIA_TXT_REL_TRANSLATOR_NEURAL_NETWORK_SEQUENCE_GRAMMAR development history for meaning and algorithmic implications of this feature
			if(resetSequentialInputsTContiguity):
				if(allowMultipleContributingSubinputsPerSequentialInput):
					averageTimeChangeOfNewInputRequiredForReset = 1
		doNotResetNeuronOutputUntilAllSequentialInputsActivated = True	#delays disabling neuron output while measuring new sequential inputs

	if(allowMultipleSubinputsPerSequentialInput):
		if(allowMultipleContributingSubinputsPerSequentialInput):
			useSparseTensors = False	#optional	#orig: False
		else:
			useSparseTensors = True		#optional
	else:
		useSparseTensors = True	#mandatory	#sparse tensors are used
		
	performTindependentFunctionOfSubInputs = False	#initialise
	if(allowMultipleSubinputsPerSequentialInput):
		if(useTcontiguity):
			performTindependentFunctionOfSubInputs = True
			
elif(algorithmSANI == "repeatedModules"): 	
	if(useMultipleSubinputsPerSequentialInput):
		allowMultipleSubinputsPerSequentialInput = True
	useSparseTensors = True	#OLD: mandatory
	inputNumberFeaturesForCurrentWordOnly = False
	
	performTindependentFunctionOfSubInputs = False	#initialise
	if(allowMultipleSubinputsPerSequentialInput):
		if(useTcontiguity):
			performTindependentFunctionOfSubInputs = True	#CHECKTHIS (implemented?)

if(useMultipleSubinputsPerSequentialInput):
	layerSizeConvergence = False #OLD: True	#CHECKTHIS
else:
	layerSizeConvergence = False

#set parameters oneSequentialInputHasOnlyOneSubinput:
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	oneSequentialInputHasOnlyOneSubinput = False	
	#maxNumberSubinputsPerSequentialInput = -1	#NA	#will depend on number neurons per layer
elif(algorithmSANI == "sharedModulesBinary"):		
	if(layerSizeConvergence):
		#if(numberOfSequentialInputs == 2):
		oneSequentialInputHasOnlyOneSubinput = True	#conditional probability determination of events
	else:
		oneSequentialInputHasOnlyOneSubinput = False
	if(allowMultipleSubinputsPerSequentialInput):	
		if(layerSizeConvergence):
			maxNumberSubinputsPerSequentialInput = 50	#~approx equal number of prev layer neurons/2 (FUTURE: make dynamic based on layer index)	#number of prior/future events in which to calculate a conditional probability	#if oneSequentialInputHasOnlyOneSubinput: can be higher because number of permutations of subinputs is lower
		else:
			maxNumberSubinputsPerSequentialInput = numberSubinputsPerSequentialInputDefault	#sparsity	#OLD: 1
elif(algorithmSANI == "sharedModules"):
	if(useSparseTensors):	#FUTURE: upgrade code to remove this requirement
		if(allowMultipleSubinputsPerSequentialInput):
			#if(numberOfSequentialInputs == 2):
			oneSequentialInputHasOnlyOneSubinput = False	#optional	#orig: True	#conditional probability determination of events
		else:
			oneSequentialInputHasOnlyOneSubinput = False
	else:
		oneSequentialInputHasOnlyOneSubinput = False
	if(allowMultipleSubinputsPerSequentialInput):
		if(useSparseTensors):
			if(oneSequentialInputHasOnlyOneSubinput):
				maxNumberSubinputsPerSequentialInput = 50	#~approx equal number of prev layer neurons/2	#number of prior/future events in which to calculate a conditional probability	#if oneSequentialInputHasOnlyOneSubinput: can be higher because number of permutations of subinputs is lower
			else:
				maxNumberSubinputsPerSequentialInput = numberSubinputsPerSequentialInputDefault	#sparsity
elif(algorithmSANI == "repeatedModules"):
	oneSequentialInputHasOnlyOneSubinput = False	#mandatory (repeatedModules does not currently support oneSequentialInputHasOnlyOneSubinput as numberSubinputsPerSequentialInput is not always calculated via calculateNumberSubinputsPerSequentialInputSparseTensors function)
	if(allowMultipleSubinputsPerSequentialInput):
		if(useSparseTensors):
			maxNumberSubinputsPerSequentialInput = numberSubinputsPerSequentialInputDefault	#sparsity
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
	
#set parameters performFunctionOfSubInputsWeighted/useLastSequentialInputOnly/numberOfWordsInConvolutionalWindowSeen:
performFunctionOfSubInputsWeighted = False 	#initialise
if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
	performThresholdOfSubInputsNonlinear = True	#default: apply non-linear activation function to sequential input (use Aseq)
	if(not performThresholdOfSubInputsNonlinear):
		performThresholdOfSubInputsBinary = True		#optional	#simple thresholding function (activations are normalised to 1.0)
		if(not performThresholdOfSubInputsBinary):
			sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)	#should this be used with backprop?
	sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)	#should this be used with backprop?
		
	performFunctionOfSubInputsWeighted = True	#mandatory (implied)
	if(performThresholdOfSubInputsNonlinear):
		performFunctionOfSubInputsNonlinear = True		#default: apply non-linear activation function to sequential input (use Aseq)
	else:
		performFunctionOfSubInputsNonlinear = False
	
	performFunctionOfSequentialInputs = True	#optional (else just take the last Zseq/Aseq values)
	if(performFunctionOfSequentialInputs):
		performFunctionOfSequentialInputsWeighted = False	#optional: multiply sequential input activations (Zseq/Aseq) by a matrix - otherwise just sum/average them together (or take a pass condition)
		if(performFunctionOfSequentialInputsWeighted):
			sequentialInputCombinationModeSummationAveraged = False	#mandatory
			if(performFunctionOfSubInputsNonlinear):
				performFunctionOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Aseq multiplied by a matrix
			else:
				performFunctionOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Zseq multiplied by a matrix
		else:
			sequentialInputCombinationModeSummationAveraged = True	#default: average values across seq (else sum)
			if(performFunctionOfSubInputsNonlinear):
				performFunctionOfSequentialInputsNonlinear = False	#default: !apply non-linear activation function neuron: use A value from typically all Aseq summed/averaged			
			else:
				performFunctionOfSequentialInputsNonlinear = True 	#default: apply non-linear activation function neuron: use Z value from typically all Zseq summed/averaged
	else:
		pass	#if(performSummationOfSubInputsBinary): simple thresholding function (activations are normalised to 1.0)
elif(algorithmSANI == "sharedModulesBinary"):		
	performFunctionOfSubInputsWeighted = False	#mandatory
	useLastSequentialInputOnly = True	#implied variable (not used)
	performFunctionOfSequentialInputsWeighted = False	#mandatory (implied)
elif(algorithmSANI == "sharedModules"):
	performThresholdOfSubInputsNonlinear = True	#default: apply non-linear activation function to sequential input (use Aseq)
	if(not performThresholdOfSubInputsNonlinear):
		sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)	#should this be used with backprop?
		
	if(allowMultipleSubinputsPerSequentialInput):
	
		if(useSparseTensors):
			if(performTindependentFunctionOfSubInputs):
				if(allowMultipleContributingSubinputsPerSequentialInput):
					#[multiple contributing subinputs per sequential input] #each sequential input can detect a pattern of activation from the previous layer
					performFunctionOfSubInputsWeighted = True	#mandatory?
				else:
					performFunctionOfSubInputsWeighted = False	#will take (True: most weighted) (False: any) active time contiguous subinput
				performFunctionOfSubInputsSummation = True
				#performFunctionOfSubInputsMax = True	#take sub input with max input signal*weight
			else:
				performFunctionOfSubInputsWeighted = True	#mandatory	#determines if backprop is required to update weight matrix associated with inputs to a sequential input?
		else:
			performFunctionOfSubInputsWeighted = True
			#performFunctionOfSubInputsAverage = True
				
		if(performThresholdOfSubInputsNonlinear):
			performFunctionOfSubInputsNonlinear = True
		else:
			performFunctionOfSubInputsNonlinear = False		

		performFunctionOfSequentialInputs = True	#optional (else just take the last Zseq/Aseq values)
		if(performFunctionOfSequentialInputs):
			performFunctionOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
			performFunctionOfSequentialInputsWeighted = False	#optional: multiply sequential input activations (Zseq/Aseq) by a matrix - otherwise just sum/average them together (or take a pass condition)
			if(performFunctionOfSequentialInputsWeighted):
				sequentialInputCombinationModeSummationAveraged = False	#mandatory
				if(performFunctionOfSubInputsNonlinear):
					performFunctionOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Aseq multiplied by a matrix
				else:
					performFunctionOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Zseq multiplied by a matrix
			else:
				sequentialInputCombinationModeSummationAveraged = True	#default: average values across seq (else sum)
				if(performFunctionOfSubInputsNonlinear):
					performFunctionOfSequentialInputsNonlinear = False	#default: !apply non-linear activation function neuron: use A value from typically all Aseq summed/averaged			
				else:
					performFunctionOfSequentialInputsNonlinear = True 	#default: apply non-linear activation function neuron: use Z value from typically all Zseq summed/averaged
		else:
			useLastSequentialInputOnly = True	#implied variable (not used)

	else:
		performFunctionOfSubInputsWeighted = False

		#required irrespective of performFunctionOfSubInputsSummation since performFunctionOfSequentialInputs code tests for performFunctionOfSubInputsNonlinear instead of performThresholdOfSubInputsNonlinear:
		if(performThresholdOfSubInputsNonlinear):
			performFunctionOfSubInputsNonlinear = True	
		else:
			performFunctionOfSubInputsNonlinear = False
				
		performFunctionOfSequentialInputs = True
		if(performFunctionOfSequentialInputs):
			performFunctionOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
			performFunctionOfSequentialInputsWeighted = True	#optional: multiply sequential input activations (Zseq/Aseq) by a matrix - otherwise just sum/average them together (or take a pass condition)
			if(performFunctionOfSequentialInputsWeighted):
				sequentialInputCombinationModeSummationAveraged = False	#mandatory
				performFunctionOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Zseq multiplied by a matrix
			else:
				sequentialInputCombinationModeSummationAveraged = True	#default: average values across seq (else sum)
				performFunctionOfSequentialInputsNonlinear = True 	#default: apply non-linear activation function neuron: use Z value from typically all Zseq summed/averaged
		else:
			useLastSequentialInputOnly = True	#implied variable (not used)
elif(algorithmSANI == "repeatedModules"): 
	performThresholdOfSubInputsNonlinear = True	#default: apply non-linear activation function to sequential input (use Aseq)
	if(not performThresholdOfSubInputsNonlinear):
		sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)	#should this be used with backprop?
		
	performFunctionOfSequentialInputsWeighted = False	#initialise
	if(allowMultipleSubinputsPerSequentialInput):
		#[multiple subinputs per sequential input] #each sequential input can detect a pattern of activation from the previous layer

		performIndependentSubInputValidation = False	#initialisation
		if(useSparseTensors):
			if(performTindependentFunctionOfSubInputs):
				performIndependentSubInputValidation = True	#performIndependentSubInputValidation is only coded for performTindependentFunctionOfSubInputs
				performFunctionOfSubInputsWeighted = True	#determines if backprop is required to update weight matrix associated with inputs to a sequential input?
				performFunctionOfSubInputsSummation = True
				#performFunctionOfSubInputsMax = True	#take sub input with max input signal*weight
			else:
				performFunctionOfSubInputsWeighted = True	#mandatory
		else:
			performFunctionOfSubInputsWeighted = True
			#performFunctionOfSubInputsAverage = True
			
		if(performThresholdOfSubInputsNonlinear):
			performFunctionOfSubInputsNonlinear = True
		else:
			performFunctionOfSubInputsNonlinear = False	

		performFunctionOfSequentialInputs = True	#optional (else just take the last Zseq/Aseq values)
		if(performFunctionOfSequentialInputs):
			performFunctionOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
			performFunctionOfSequentialInputsWeighted = False	#optional: multiply sequential input activations (Zseq/Aseq) by a matrix - otherwise just sum/average them together (or take a pass condition)
			if(performFunctionOfSequentialInputsWeighted):
				sequentialInputCombinationModeSummationAveraged = False	#mandatory
				if(performFunctionOfSubInputsNonlinear):
					performFunctionOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Aseq multiplied by a matrix
				else:
					performFunctionOfSequentialInputsNonlinear = True	#default: apply non-linear activation function neuron: use A value from typically all Zseq multiplied by a matrix
			else:
				sequentialInputCombinationModeSummationAveraged = True	#default: average values across seq (else sum)
				if(performFunctionOfSubInputsNonlinear):
					performFunctionOfSequentialInputsNonlinear = False	#default: !apply non-linear activation function neuron: use A value from typically all Aseq summed/averaged			
				else:
					performFunctionOfSequentialInputsNonlinear = True 	#default: apply non-linear activation function neuron: use Z value from typically all Zseq summed/averaged
		else:
			useLastSequentialInputOnly = True	#implied variable (not used)
		
		if(enforceTcontiguityConstraints):
			sequentialityMode = "default"
			#sequentialityMode = "temporalCrossoverAllowed"
			#sequentialityMode = "contiguousInputEnforced"
	else:
		#[single subinput per sequential input] #each sequential input is directly connected to a single neuron on the previous layer

		performIndependentSubInputValidation = False	#always False (ie false by definition because there is only 1 subinput per sequential input)

		performFunctionOfSequentialInputs = True
		if(performFunctionOfSequentialInputs):
			performFunctionOfSequentialInputsWeighted = True	#does backprop require to update weight matrix associated with sequential inputs?
			performFunctionOfSequentialInputsNonlinear = True
		else:
			performFunctionOfSequentialInputsWeighted = False
			performFunctionOfSequentialInputsNonlinear = False

		if(enforceTcontiguityConstraints):
			sequentialityMode = "default"
			#sequentialityMode = "temporalCrossoverAllowed"
			#sequentialityMode = "contiguousInputEnforced"

		
#set parameters record:
recordNetworkWeights = False	#initialise
if(not useLearningRuleBackpropagation):		
	#rationale: prune network neurons/connections based on the relative strength of these weights
	#need to modify weights not just record them
	if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):	
		recordNetworkWeights = False	#not available
	elif(algorithmSANI == "sharedModulesBinary"):
		if(not ANNtf2_globalDefs.testHarness):	
			recordNetworkWeights = True
			if(recordNetworkWeights):
				recordSubInputsWeighted = True
				recordSequentialInputsWeighted = False	#may not be necessary (only used if can split neuron sequential inputs)
				recordNeuronsWeighted = True
	elif(algorithmSANI == "sharedModules"):
		if(useSparseTensors):
			if(allowMultipleSubinputsPerSequentialInput):
				recordNetworkWeights = True
				if(recordNetworkWeights):
					recordSubInputsWeighted = True
					recordSequentialInputsWeighted = False	#may not be necessary (only used if can split neuron sequential inputs)
					recordNeuronsWeighted = True
			else:
				recordNetworkWeights = False	#not yet coded
		else:
			recordNetworkWeights = False	#not yet coded
	elif(algorithmSANI == "repeatedModules"): 	
		recordNetworkWeights = False	#not available


supportSkipLayers = False	#initialise
if(useSkipLayers):
	if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
		supportSkipLayers = True
	elif(algorithmSANI == "sharedModulesBinary"):
		supportSkipLayers = True
	elif(algorithmSANI == "sharedModules"):
		if(allowMultipleSubinputsPerSequentialInput):
			if(useSparseTensors):
				supportSkipLayers = True
			else:
				supportSkipLayers = True
		else:
			supportSkipLayers = True
	elif(algorithmSANI == "repeatedModules"):
		supportSkipLayers = True

	
#if(algorithmSANI == "sharedModulesNonContiguousFullConnectivity"):
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


#def setNumberOfWordsInConvolutionalWindowSeenFromDatasetPOStagSequence(n):
#	global numberOfWordsInConvolutionalWindowSeen
#	numberOfWordsInConvolutionalWindowSeen = n

def getNumberOfWordsInConvolutionalWindowSeenFromDatasetPOStagSequence(dataset, num_input_neurons, numberOfFeaturesPerWord):
	if(dataset == "POStagSequence" and not SANIsharedModules):
		return num_input_neurons//numberOfFeaturesPerWord
	else:
		return numberOfWordsInConvolutionalWindowSeen
		
	
		
