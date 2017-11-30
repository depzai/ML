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
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

# Now the model is done we can train this model. We have to explain
# to tensorflow what to do with it
# we need to run data through the model in the sessions


def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)











