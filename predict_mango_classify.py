import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import collections
import time
import threading
import logging
import sys
import argparse

__author__ = 'Ryan Joshua Liwag'

def get_args():
    parser = argparse.ArgumentParser(
        description='Script recieves input image and model locations to classify a single image')
    # Add arguments
    parser.add_argument(
        '-i', '--image', type=str, help='image file location', required=True)
    parser.add_argument(
        '-m1', '--mtl', type=str, help='MTL model location', required=True)
    parser.add_argument(
        '-m2', '--ssd', type=str, help='SSD model location', required=True)

    args = parser.parse_args()

    image = args.image
    mtl = args.mtl
    ssd = args.ssd

    return image, mtl, ssd

# Prevent use of GPU and only use CPU
config = tf.ConfigProto(
    device_count={'GPU': 0})

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in the graph
        tf.import_graph_def(graph_def, name="prefix")
    return graph

class Import_Frcnn():
	'''
	Imports frozen graph of the Single-Shot Detector Model
	'''
	def __init__(self, location):
		self.graph_frcnn = tf.Graph()
		self.sess = tf.Session(graph=self.graph_frcnn, config=config)
		with self.graph_frcnn.as_default():
			self.od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(location, 'rb') as fid:
				serialized_graph = fid.read()
				self.od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(self.od_graph_def, name='')

		try:
			self.image_tensor = self.graph_frcnn.get_tensor_by_name('image_tensor:0')
			self.detection_boxes = self.graph_frcnn.get_tensor_by_name('detection_boxes:0')
			self.detection_scores = self.graph_frcnn.get_tensor_by_name('detection_scores:0')
			self.detection_classes = self.graph_frcnn.get_tensor_by_name('detection_classes:0')
			self.num_detections = self.graph_frcnn.get_tensor_by_name('num_detections:0')
		except:
			logging.warning("Failed to Load SSD graph")

	def run(self, frame):
		image_np = np.expand_dims(frame, axis=0)
		return self.sess.run([self.detection_boxes,
							  self.detection_scores, 
							  self.detection_classes, 
							  self.num_detections],
							  feed_dict={self.image_tensor: image_np})

class Import_MTL():

	def __init__(self, location):
		try:
			self.graph_mtl = load_graph(location)
			self.sess = tf.Session(graph=self.graph_mtl, config=config)
			self.y_pred_quality = self.graph_mtl.get_tensor_by_name("prefix/y_pred_quality:0")
			self.y_pred_ripeness = self.graph_mtl.get_tensor_by_name("prefix/y_pred_ripeness:0")
			self.x = self.graph_mtl.get_tensor_by_name("prefix/x:0") 
		except:
			logging.warning("Failed to Load MTL graph")

	def run(self, frame):
		frame = Image.fromarray(np.uint8(frame*255))
		image_rgb= frame.resize((50,50), Image.ANTIALIAS)
		image_rgb = np.expand_dims(image_rgb, axis=0)
		return self.sess.run([self.y_pred_quality,
							 self.y_pred_ripeness],
							 feed_dict={self.x: image_rgb})

def get_box(boxes, scores, image):

	boxes = np.squeeze(boxes)
	height, width = image.shape[:2]
	box = None
	score = None

	# Get only objects that have an accuracy greater than 80%
	if scores.item(0) > 0.8:
		ymin, xmin, ymax, xmax = boxes[0]
		box = [xmin * width, xmax * width, ymin * height, ymax * height]
		score = scores.item(0)
	else: 
		pass

	return box, score

def draw_boxes_scores(frame, box_array, score_array, ripe_array, quality_array):
	ripeness_dict = {0: 'Green', 1: 'Semi-Ripe', 2: 'Ripe'}
	quality_dict = {0: 'Good', 1: 'Defect'}
	frame = Image.fromarray(frame.astype('uint8'), 'RGB')
	draw = ImageDraw.Draw(frame)
	draw.rectangle([(int(box_array[0]), int(box_array[2])), (int(box_array[1]), int(box_array[3]))], fill=None, outline="blue")
	draw.text((int(box_array[0]),int(box_array[2]-17)), "Detection:{0:.2f}".format(score_array), 
			   font= ImageFont.truetype("arial.ttf", 15), fill="blue")
	draw.text((int(box_array[0]),int(box_array[2]-30)), "Quality:{}".format(quality_dict[int(np.argmax(quality_array, axis=1))]), 
	   		   font= ImageFont.truetype("arial.ttf", 15), fill="blue")
	draw.text((int(box_array[0]),int(box_array[2]-43)), "Ripeness:{}".format(ripeness_dict[int(np.argmax(ripe_array, axis=1))]), 
	  		   font= ImageFont.truetype("arial.ttf", 15), fill="blue")
	frame.show()
	frame.save("tmp.png")

class MyThread(threading.Thread):

	def __init__(self, ssd_location, mtl_location):
		threading.Thread.__init__(self)
		self.ssd_location = ssd_location
		self.mtl_location = mtl_location
		self.stop = threading.Event()
		self.create_models()

	def run(self, frame):
		self.predict(frame)
		
	def predict(self, frame):
		(boxes, scores, classes, num) = self.model_fcnn.run(frame)
		box_array, scores_ = get_box(boxes, scores, frame)
		
		if scores_:
			left, right, top, bottom  = box_array
			crop = frame[int(top):int(bottom), int(left):int(right)]
			quality, ripeness = self.model_mtl.run(crop)
			draw_boxes_scores(frame, box_array, scores_, ripeness, quality)
		else:
			pass

	def create_models(self):
		'''
		Initializes both Frcnn and MTL model architectures
		'''
		self.model_mtl = Import_MTL(self.mtl_location)
		self.model_fcnn = Import_Frcnn(self.ssd_location)

	def terminate(self):
		self.stop.set()

def main():
	image, mtl, ssd = get_args()
	img = Image.open(image)

	thread_1 = MyThread(ssd, mtl)
	img = np.array(img)
	thread_1.run(img)
	
	sys.exit()

if __name__ == "__main__":
	main()


