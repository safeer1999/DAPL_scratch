import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#Datasets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Paramters
learning_rate = 0.01
epochs = 10
batch_size = 150

#training data placeholders
X = tf.placeholder(tf.float32, [None,784]) #input
initializer=tf.variance_scaling_initializer()

#Weights and bias
W1 = tf.Variable(initializer([784,392]), name = 'W1')
b1 = tf.Variable(tf.zeros([392]), name = 'b1')

W2 = tf.Variable(initializer([392,196]), name = 'W2')
b2 = tf.Variable(tf.zeros([196]), name = 'b2')

W3 = tf.Variable(initializer([196,392]), name = 'W3')
b3 = tf.Variable(tf.zeros([392]), name = 'b3')

W4 = tf.Variable(initializer([392,784]), name = 'W4')
b4 = tf.Variable(tf.zeros([784]), name = 'b4')

#Calculating nodes values
enc_1 = tf.add(tf.matmul(X,W1),b1)
enc_1 = tf.nn.relu(enc_1)

enc_2 = tf.add(tf.matmul(enc_1,W2),b2)
enc_2 = tf.nn.relu(enc_2)


dec_1 = tf.add(tf.matmul(enc_2,W3),b3)
dec_1 = tf.nn.relu(dec_1)

dec_2 = tf.add(tf.matmul(dec_1,W4),b4)
dec_2 = tf.nn.relu(dec_2)

# Reconstruction Output
y_pred = dec_2


#Loss functions
loss=tf.reduce_mean(tf.square(y_pred-X))

print(y_pred)
print(X)

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#Initialization
init_op = tf.global_variables_initializer()

#Running session - training

with tf.Session() as sess :

	sess.run(init_op)
	total_batch = int(len(mnist.train.labels) / batch_size)

	for epoch in range(epochs) :

		l = 0
		for i in range(total_batch) :

			batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)

			#print("X: ",type(batch_x)," ",batch_x.shape,"\n\n")
			#print("y: ",type(batch_y)," ",batch_y.shape,"\n\n\n")


			_, l = sess.run([optimizer, loss], feed_dict = {X : batch_x})

		 

		print("Epoch: ", epoch + 1, "cost: ", "{:.3f}".format(l))



