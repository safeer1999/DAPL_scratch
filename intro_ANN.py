import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#Datasets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Optimization Variables
learning_rate = 0.5
epochs = 10
batchsize = 100

#training data placeholders
x = tf.placeholder(tf.float32, [None,784]) #input
y = tf.placeholder(tf.float32, [None,10]) #output

#Weights and Biases
W1 = tf.Variable(tf.random_normal([784,300], stddev = 0.03), name = 'W1')
b1 = tf.Variable(tf.random_normal([300]), name = 'b1')

W2 = tf.Variable(tf.random_normal([300,10], stddev = 0.03), name = 'W2')
b2 = tf.Variable(tf.random_normal([10]), name = 'b2')


#Calculating the values of nodes in hidden layer
hidden_out = tf.add(tf.matmul(x,W1), b1)
hidden_out = tf.nn.relu(hidden_out)

#Calcuating output layer nodes values
output_layer = tf.add(tf.matmul(hidden_out,W2),b2)
y_ = tf.nn.softmax(output_layer)

#Loss functions
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

#Optimizer
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


#initialization
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#running the session- training the data
with tf.Session() as sess :

	sess.run(init_op)
	total_batch = int(len(mnist.train.labels) / batchsize)

	for epoch in range(epochs) :

		avg_cost = 0
		for i in range(total_batch) :

			batch_x, batch_y = mnist.train.next_batch(batch_size = batchsize)

			print("X: ",type(batch_x)," ",batch_x.shape,"\n\n")
			print("y: ",type(batch_y)," ",batch_y.shape,"\n\n\n")


			_, c = sess.run([optimiser, cross_entropy], feed_dict = {x : batch_x, y : batch_y})

			avg_cost += c 

		print("Epoch: ", epoch + 1, "cost: ", "{:.3f}".format(avg_cost))

	
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

