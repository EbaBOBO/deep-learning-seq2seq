import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
		###### DO NOT CHANGE ##############
		super(RNN_Seq2Seq, self).__init__()
		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters

		# Define batch size and optimizer/learning rate
		self.batch_size = 100 # You can change this
		self.embedding_size = 100 # You should change this
	
		# 2) Define embeddings, encoder, decoder, and feed forward layers
		self.embed_eng = tf.Variable(tf.random.truncated_normal([self.english_vocab_size,self.embedding_size], stddev=0.01))
		self.embed_fre = tf.Variable(tf.random.truncated_normal([self.french_vocab_size,self.embedding_size], stddev=0.01))
		self.LSTM_encoder = tf.keras.layers.LSTM(50,return_sequences=True, return_state=True)
		self.LSTM_decoder = tf.keras.layers.LSTM(50,return_sequences=True, return_state=True)
		self.dense_layer1 = tf.keras.layers.Dense(100)
		self.dense_layer2 = tf.keras.layers.Dense(self.english_vocab_size)


	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
	
		# TODO:
		#1) Pass your french sentence embeddings to your encoder 
		embed_french = tf.nn.embedding_lookup(self.embed_fre, encoder_input, max_norm=None, name=None)
		encoder,final_memory_state, final_carry_state= self.LSTM_encoder(embed_french)
		
		#2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
		embed_eng = tf.nn.embedding_lookup(self.embed_eng, decoder_input, max_norm=None, name=None)
		decoder, final_memory_state, final_carry_state = self.LSTM_encoder(embed_eng, [final_memory_state,final_carry_state])
		#3) Apply dense layer(s) to the decoder out to generate probabilities
		logits1 = self.dense_layer1(decoder)
		logits2 = self.dense_layer2(logits1)
		prbs = tf.nn.softmax(logits2)
		return prbs

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the total model cross-entropy loss after one forward pass. 
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		losses = tf.keras.metrics.sparse_categorical_crossentropy(labels, prbs)
		return tf.reduce_sum(tf.boolean_mask(losses,mask))

