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
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100































