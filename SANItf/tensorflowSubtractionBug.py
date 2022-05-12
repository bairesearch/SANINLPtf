"""tensorflowSubtractionBug.py
"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

randomNormal = tf.initializers.RandomNormal()

Z = randomNormal([2, 2], dtype=tf.float32)
	#tf.zeros([2, 2], dtype=tf.float32)	#tensor
a = 0.0	#python

#Eager execution of this command throws a non-deterministic error (corrupts subsequent tensor output);

Z = Z-a
#[this is OK: Z = Z - a]

print("Z = ", Z)
					
