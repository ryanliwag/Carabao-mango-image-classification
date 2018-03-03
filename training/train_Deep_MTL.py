from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from random import shuffle
import pandas as pd

import tensorflow as tf



def read_data(data):
    dataset = []
    label1 = []
    label2 = []
    for i in data.values:
        dataset.append(i[0])
        
        if i[1] == "good":
            label1.append(0)
        elif i[1] == "defect":
            label1.append(1)
            
        if i[2] == "green":
            label2.append(0)
        elif i[2] == "semi-ripe":
            label2.append(1)
        else:
            label2.append(2)
        
    return dataset, label1, label2

def _parse_function(filename, label_quality, label_ripeness):

  one_hot_quality = tf.one_hot(label_quality, 2)
  one_hot_ripeness = tf.one_hot(label_ripeness, 3)

  img_file = tf.read_file(filename)
  img_decoded = tf.image.decode_jpeg(img_file, channels=3)
  image_resized = tf.image.resize_images(img_decoded, [50, 50])
  return image_resized, one_hot_quality, one_hot_ripeness


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


#Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 64

filter_size_conv3 = 3
num_filters_conv3 = 128

filter_size_conv4 = 3
num_filters_conv4 = 256
    
fc_layer_size = 256

fc_layer_size2 = 128

#model params
img_size=50
num_channels=3
num_classes_ripeness = 3
num_classes_quality = 2

# input
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

layer_conv4= create_convolutional_layer(input=layer_conv3,
               num_input_channels=num_filters_conv3,
               conv_filter_size=filter_size_conv4,
               num_filters=num_filters_conv4)
          
shared_flatten_layer = create_flatten_layer(layer_conv4)


layer_fc1_ripeness = create_fc_layer(input=shared_flatten_layer,
                     num_inputs=shared_flatten_layer.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2_ripeness = create_fc_layer(input=layer_fc1_ripeness,
                     num_inputs=fc_layer_size,
                     num_outputs=fc_layer_size2,
                     use_relu=True) 

layer_fc3_ripeness = create_fc_layer(input=layer_fc2_ripeness,
                     num_inputs=fc_layer_size2,
                     num_outputs=num_classes_ripeness,
                     use_relu=False) 


layer_fc1_quality = create_fc_layer(input=shared_flatten_layer,
                     num_inputs=shared_flatten_layer.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2_quality = create_fc_layer(input=layer_fc1_quality,
                     num_inputs=fc_layer_size,
                     num_outputs=fc_layer_size2,
                     use_relu=True) 

layer_fc3_quality = create_fc_layer(input=layer_fc2_quality,
                     num_inputs=fc_layer_size2,
                     num_outputs=num_classes_quality,
                     use_relu=False) 



#labels ripeness
y_true_ripeness = tf.placeholder(tf.float32, shape=[None, num_classes_ripeness], name='y_true_ripe')
y_true_cls_ripeness = tf.argmax(y_true_ripeness, axis=1)

#labels quality
y_true_quality = tf.placeholder(tf.float32, shape=[None, num_classes_quality], name='y_true_quality')
y_true_cls_quality = tf.argmax(y_true_quality, axis=1)

#prediction
y_pred_ripeness = tf.nn.softmax(layer_fc3_ripeness,name='y_pred_ripeness')
y_pred_cls_ripeness = tf.argmax(y_pred_ripeness, axis=1)

y_pred_quality = tf.nn.softmax(layer_fc3_quality,name='y_pred_quality')
y_pred_cls_quality = tf.argmax(y_pred_quality, axis=1)


#cost function
cross_entropy_ripeness = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3_ripeness,labels=y_true_ripeness)
cross_entropy_quality = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3_quality,labels=y_true_quality)
cost_ripeness = tf.reduce_mean(cross_entropy_ripeness)
cost_quality = tf.reduce_mean(cross_entropy_quality)
combined_cost = tf.add(cost_ripeness, cost_quality)

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(combined_cost)

#accuracy
correct_prediction_ripeness = tf.equal(y_pred_cls_ripeness, y_true_cls_ripeness)
accuracy_ripeness = tf.reduce_mean(tf.cast(correct_prediction_ripeness, tf.float32))

correct_prediction_quality = tf.equal(y_pred_cls_quality, y_true_cls_quality)
accuracy_quality = tf.reduce_mean(tf.cast(correct_prediction_quality, tf.float32))



# Loading Data
data_full=pd.read_csv("/home/elements/Desktop/v-env/tensorflow/My-Tensorflow-Basics/object-detection/final_form.csv")
data_rand = data_full.sample(frac=1)

dataset_train = data_rand[:5800]
dataset_test = data_rand[5800:]

train, quality_label, ripeness_label = read_data(dataset_train)
test, test_quality_label, test_ripeness_label = read_data(dataset_test)

training_files = tf.constant(train)
training_label_quality = tf.constant(quality_label)
training_label_ripeness = tf.constant(ripeness_label)
testing_files = tf.constant(test)
testing_label_quality = tf.constant(test_quality_label)
testing_label_ripeness = tf.constant(test_ripeness_label)

# create TensorFlow Dataset objects
tr_data = tf.contrib.data.Dataset.from_tensor_slices((training_files, training_label_quality, training_label_ripeness))
tr_data = tr_data.map(_parse_function)
tr_data = tr_data.shuffle(buffer_size=10000)
tr_data = tr_data.repeat()
tr_data = tr_data.batch(100)


val_data = tf.contrib.data.Dataset.from_tensor_slices((testing_files, testing_label_quality, testing_label_ripeness))
val_data = val_data.map(_parse_function)
val_data = val_data.shuffle(buffer_size=10000)
val_data = val_data.repeat()
val_data = val_data.batch(100)



# create TensorFlow Iterator object
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.contrib.data.Iterator.from_string_handle(
    handle, tr_data.output_types, tr_data.output_shapes)
next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = tr_data.make_initializable_iterator()
validation_init_op = val_data.make_initializable_iterator()



saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  handle_train, handle_val = sess.run([training_init_op.string_handle(), validation_init_op.string_handle()])

  for i in range(0,800):
    sess.run(training_init_op.initializer)
    tr_elem = sess.run(next_element, feed_dict={handle: handle_train})
    sess.run(optimizer, feed_dict={x: tr_elem[0], y_true_quality: tr_elem[1], y_true_ripeness: tr_elem[2]})
    

    if i % int(len(train)/100) == 0:
      sess.run(validation_init_op.initializer)
      vl_elem = sess.run(next_element, feed_dict={handle: handle_val})

      #train_acc_cost
      train_acc_ripeness, train_acc_quality = sess.run([accuracy_ripeness, accuracy_quality], feed_dict={x:tr_elem[0], y_true_quality: tr_elem[1], y_true_ripeness: tr_elem[2]})
      total_loss = sess.run(combined_cost, feed_dict={x:tr_elem[0], y_true_quality: tr_elem[1], y_true_ripeness: tr_elem[2]})

      #val_acc_cost 
      val_acc_ripeness, val_acc_quality = sess.run([accuracy_ripeness, accuracy_quality], feed_dict={x:vl_elem[0], y_true_quality: vl_elem[1], y_true_ripeness: vl_elem[2]})
      val_total_loss = sess.run(combined_cost, feed_dict={x:vl_elem[0], y_true_quality: vl_elem[1], y_true_ripeness: vl_elem[2]})

      epoch = int(i / int(len(train)/100))  

      msg_train= "Training Epoch {0} --- Training Accuracy Ripeness: {1:>6.1%}, Training Accuracy Quality: {2:>6.1%},  Total Training Loss: {3:.3f}"
      msg_val = "Training Epoch {0} --- Validation Accuracy Ripeness: {1:>6.1%}, Validation Accuracy Quality: {2:>6.1%},  Total Validation Loss: {3:.3f}"

      print(msg_train.format(epoch + 1, train_acc_ripeness, train_acc_quality, total_loss))
      print(msg_val.format(epoch + 1, val_acc_ripeness, val_acc_quality, val_total_loss))


saver.save(sess, 'test4-model') 

  