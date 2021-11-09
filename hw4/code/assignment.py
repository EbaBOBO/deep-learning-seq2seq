import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random


from attenvis import AttentionVis
av = AttentionVis()

def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
	# and english sentences to be used by the decoder
	# - The english sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
	# 
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP] 


	# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
	for i in range(int(len(train_french)/model.batch_size)):
		with tf.GradientTape() as tape:
			train_french1 = train_french[i*model.batch_size:(i+1)*model.batch_size]
			train_english1 = train_english[i*model.batch_size:(i+1)*model.batch_size,:-1]
			probs=model.call(train_french1,train_english1)
            # print(probs.shape)
			label = train_english[i*model.batch_size:(i+1)*model.batch_size,1:]
			losses=model.loss_function(probs,label,label != eng_padding_index)
		gradients = tape.gradient(losses, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	pass

@av.test_func
def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
	e.g. (my_perplexity, my_accuracy)
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!
	sumper = 0
	sumacc = 0
	total=0
	for i in range(int(len(test_french)/model.batch_size)):
		# with tf.GradientTape() as tape:
		test_french1 = test_french[i*model.batch_size:(i+1)*model.batch_size]
		test_english1 = test_english[i*model.batch_size:(i+1)*model.batch_size,:-1]
		probs=model.call(test_french1,test_english1)
		# print(probs.shape)
		label = test_english[i*model.batch_size:(i+1)*model.batch_size,1:]
		losses=model.loss_function(probs,label, label != eng_padding_index)
		sumper += losses
		acc = model.accuracy_function(probs,label, label != eng_padding_index)
		sumacc += np.sum(label != eng_padding_index)*acc
		total += np.sum(label != eng_padding_index)
	perplexity = np.exp(sumper/total)
	accuracy = sumacc/total
	print('perplexity is:',perplexity)
	print('acc is:',accuracy)
	return perplexity,accuracy
	pass

def main():	
	if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [RNN/TRANSFORMER]")
			exit()

	# Change this to "True" to turn on the attention matrix visualization.
	# You should turn this on once you feel your code is working.
	# Note that it is designed to work with transformers that have single attention heads.
	if sys.argv[1] == "TRANSFORMER":
		av.setup_visualization(enable=False)

	print("Running preprocessing...")
	train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = get_data('../../data/fls.txt','../../data/els.txt','../../data/flt.txt','../../data/elt.txt')
	print("Preprocessing complete.")

	model_args = (FRENCH_WINDOW_SIZE, len(french_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))
	if sys.argv[1] == "RNN":
		model = RNN_Seq2Seq(*model_args)
	elif sys.argv[1] == "TRANSFORMER":
		model = Transformer_Seq2Seq(*model_args) 
	
	# TODO:
	# Train and Test Model for 1 epoch.
	train(model,train_french,train_english,eng_padding_index)
	test(model,test_french,test_english,eng_padding_index)



	# Visualize a sample attention matrix from the test set
	# Only takes effect if you enabled visualizations above
	av.show_atten_heatmap()
	pass

if __name__ == '__main__':
	main()
