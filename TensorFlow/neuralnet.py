# # tensor is an array
# #using values, but x1 and 2 can be arrays
# x1 = tf.constant(5)
# x2 = tf.constant(6)

# #this below doesn't actually triggers the calc, it sets up the model
# result = tf.multiply(x1,x2)
# print(result)

# #now run the model in a session:
# sess = tf.Session()
# print(sess.run(result))
# sess.close()

# #or
# with tf.Session() as sess:
#     output = sess.run(result)
#     print(output)

'''
MNIST Dataset is handwritten numbers: features are pixels
Principle:
input > weight > hiddenlayer 1 (activation function)> weights> hidden layer 2
(activation function) > weights > output layer
Then compare output and intended output (cost or loss function)
Optimization function (optimizer) > minimize cost (backpropagation)

feed forward + backprop = epoch: you run several epochs to bring the cost down

'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
#one_hot means one is on and the rest are off:
'''
for 10 classes:
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
'''
#3 hidden layers (deep neural network so more than one hidden layer)
# number of nodes can change
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

batch_size = 100 # goes through batches of 100 features at a time


# height by width
x = tf.placeholder('float',[None, 784]) # 784 values (28 by 28 pixels), doesn't have to be a mtrix, so no other dims (None)
y = tf.placeholder('float') # labels

# input data * weights + bias is the formula, bias in case all inputs are 0

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
	'biases':tf.variable(tf.random_normal(n_nodes_hl1))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
	'biases':tf.variable(tf.random_normal(n_nodes_hl2))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
	'biases':tf.variable(tf.random_normal(n_nodes_hl3))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl1])),
	'biases':tf.variable(tf.random_normal([n_classes]))}


	# for each layer: input data * weights + bias is the formula, bias in case all inputs are 0

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights'])) + hidden_1_layer['biases']
	#activation function (sigmoid to define neuron's value)
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights'])) + hidden_2_layer['biases']
	#activation function (sigmoid to define neuron's value)
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights'])) + hidden_3_layer['biases']
	#activation function (sigmoid to define neuron's value)
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights'])) + output_layer['biases']

	return output
















