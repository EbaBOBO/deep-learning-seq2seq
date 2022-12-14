import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis

av = AttentionVis()

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters
		# 2) Define embeddings, encoder, decoder, and feed forward layers

		# Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 100
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		# Define english and french embedding layers: 
		self.embed_eng = tf.Variable(tf.random.truncated_normal([self.english_vocab_size,self.embedding_size], stddev=0.01))
		self.embed_fre = tf.Variable(tf.random.truncated_normal([self.french_vocab_size,self.embedding_size], stddev=0.01))

		# Create positional encoder layers
		self.pos_layer1 = transformer.Position_Encoding_Layer(self.french_window_size,self.embedding_size)
		self.pos_layer2 = transformer.Position_Encoding_Layer(self.english_window_size,self.embedding_size)

		# Define encoder and decoder layers:
		self.encoder_layer = transformer.Transformer_Block(self.embedding_size,is_decoder = False)
		self.decoder_layer = transformer.Transformer_Block(self.embedding_size,is_decoder = True)
		# Define dense layer(s)
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
		#1) Add the positional embeddings to french sentence embeddings
		embed_french = tf.nn.embedding_lookup(self.embed_fre, encoder_input, max_norm=None, name=None)
		pos_french = self.pos_layer1(embed_french)
		#2) Pass the french sentence embeddings to the encoder
		encoder = self.encoder_layer(pos_french)
		#3) Add positional embeddings to the english sentence embeddings
		embed_eng = tf.nn.embedding_lookup(self.embed_eng, decoder_input, max_norm=None, name=None)
		pos_eng = self.pos_layer2(embed_eng)
		#4) Pass the english embeddings and output of your encoder, to the decoder
		decoder = self.decoder_layer(pos_eng,encoder)
		#5) Apply dense layer(s) to the decoder out to generate probabilities
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
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		# Note: you can reuse this from rnn_model.
		losses = tf.keras.metrics.sparse_categorical_crossentropy(labels, prbs)
		return tf.reduce_sum(tf.boolean_mask(losses,mask))	

	@av.call_func
	def __call__(self, *args, **kwargs):
		return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)