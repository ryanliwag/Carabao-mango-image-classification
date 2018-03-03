#!/usr/bin/python3

'''
Made by: Ryan Joshua Liwag
'''

from __future__ import print_function, division

import json
import numpy as np
import tensorflow as tf
import os

import pandas as pd
import matplotlib.pyplot as plt

logs_path = 'logs'

def load_data(location):
    data = pd.read_csv(location)
    x_data = data[["Width", "Length"]].values
    y_data = data[["Weight"]].values
    return x_data, y_data

def normalize(data):
    me = data.mean(axis=0)
    std = data.std(axis=0)
    x_data = (data - me) / std
    return x_data, me, std

def freeze_graph(model_dir, output_node_names):
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/LR_frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

class TFLinearRegression():
    def __init__(self, savefile, D=None, K=None):
        self.savefile = savefile
        if D and K:
            self.build(D,K)

    def build(self, D, K, lr, mu, reg):
        #Linear regression model is y = Wx + b
        self.x = tf.placeholder(tf.float32, [None, D], name = "x")
        self.y_ = tf.placeholder(tf.float32, [None, 1], name = "y_")
        self.W = tf.Variable(tf.zeros([D, 1]), name = "W")
        self.b = tf.Variable(tf.zeros([1]), name = "b")

        self.saver = tf.train.Saver({'W': self.W, 'b': self.b})

        with tf.name_scope("Wx_b") as scope:
            y = tf.add(tf.matmul(self.x, self.W),self.b)

        W_hist = tf.summary.histogram("weights", self.W)
        b_hist = tf.summary.histogram("biases", self.b)
        y_hist = tf.summary.histogram("y", y)



        #cost function (sum((y_-y)^2)
        with tf.name_scope("cost") as scope:
            l2_penalty = reg*tf.reduce_mean(self.W**2) / 2
            cost = tf.reduce_mean(tf.square(self.y_-y))
            cost += l2_penalty
            cost_total = tf.summary.scalar("cost", cost)

        # Gradient Descent
        with tf.name_scope("train") as scope:
            train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)

        return cost, train_op, y

    def fit(self, X, Y, Xtest, Ytest):
        N, D = X.shape
        K = len(Y)

        max_iter = 50000
        lr = 1e-5
        mu = 0.8
        regularization = 1e-1

        cost, train, y = self.build(D, K, lr, mu, regularization)
        cost_list = []
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            sess.run(init)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            for i in range(max_iter):
                sess.run(train, feed_dict = {self.x: X, self.y_: Y})
                cost_test = sess.run(cost, feed_dict={self.x: X, self.y_:Y})
                y_test = sess.run(y, feed_dict={self.x: X, self.y_:Y})
                print("Step:", i)
                print("Cost = ", cost_test)
                cost_list.append(cost_test)
                #print("Y = ", y_test)
                result = sess.run(merged, feed_dict= {self.x: X, self.y_: Y})
                writer.add_summary(result, i)

            self.saver.save(sess, self.savefile)


def main():
    if not os.path.exists("model"):
        os.makedirs("model")

    x, y = load_data("dataset/mango_sizes.csv")

    model = TFLinearRegression("model/tf.model")
    x_norm, me, std = normalize(x)
    model.fit(x_norm,y,x,y)

    print("Model has been succesfully saved")
    print("Normalization values: mean: {}, stddev: {}".format(me, std))

    freeze_graph("model/", "y")
    print("Graph has been succesfully Frozen")
    print("Log files are saved in logs/ folder")
    print("Frozen.pb file and model save in model/ folder")
    print("View graphs using: tensorboard --logdir logs ")


if __name__ == '__main__':
    main()



