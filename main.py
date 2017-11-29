## import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os


from mango_parameters import *

import tensorflow as tf
import numpy as np
import collections
from mango_parameters import *
from size_classifier import *


import threading


import time


class Import_Frcnn():
	def __init__(self, location):
		self.graph_frcnn = tf.Graph()
		self.sess = tf.Session(graph=self.graph_frcnn)
		with self.graph_frcnn.as_default():
			self.od_graph_def = tf.GraphDef()	
			with tf.gfile.GFile(location, 'rb') as fid:
				serialized_graph = fid.read()
				self.od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(self.od_graph_def, name='')

		self.image_tensor = self.graph_frcnn.get_tensor_by_name('image_tensor:0')
		self.detection_boxes = self.graph_frcnn.get_tensor_by_name('detection_boxes:0')
		self.detection_scores = self.graph_frcnn.get_tensor_by_name('detection_scores:0')
		self.detection_classes = self.graph_frcnn.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.graph_frcnn.get_tensor_by_name('num_detections:0')
		print("Model frcnn ready")

	def run(self, frame):
		image_np = np.expand_dims(frame, axis = 0)
		return self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],feed_dict={self.image_tensor: image_np})


class Import_MTL():

	def __init__(self, location):

		self.graph_mtl = load_graph(location)
		self.sess = tf.Session(graph = self.graph_mtl)
		'''
		for op in self.graph_mtl.get_operations():
			print(str(op.name)) 
		'''
		self.y_pred_quality = self.graph_mtl.get_tensor_by_name("prefix/y_pred_quality:0")
		self.y_pred_ripeness = self.graph_mtl.get_tensor_by_name("prefix/y_pred_ripeness:0")
		self.x = self.graph_mtl.get_tensor_by_name("prefix/x:0") 
		print("Model MTL ready")


	def run(self, frame):
		image_rgb = cv2.resize(frame, (50,50))
		image_rgb = np.expand_dims(image_rgb, axis = 0)
		return self.sess.run([self.y_pred_quality, self.y_pred_ripeness], feed_dict={self.x: image_rgb})


def get_box(boxes, scores, image):
	boxes = np.squeeze(boxes)
	height, width = image.shape[:2]
	box = None
	score = None

	ymin, xmin, ymax, xmax = boxes[0]
	box = [xmin * width, xmax * width, ymin * height, ymax * height]
	score = scores.item(0)

	return box, score

	
class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.stop = threading.Event()
		self.create_models()

	def run(self, frame):
		# Load the VGG16 network

		return self.predict(frame)
		

	def predict(self, frame):

		(boxes, scores, classes, num) = self.model_fcnn.run(frame)
		box_array, scores_ = get_box(boxes, scores, frame)
		if scores_:
			left, right, top, bottom  = box_array
			crop = frame[int(top):int(bottom), int(left):int(right)]
			x, y =get_size(crop, 82)
			weight, size = predict_size([x,y])

			quality, ripeness = self.model_mtl.run(crop)
			img, quality_, ripeness_ = draw_boxes_scores(box_array, scores_, ripeness, quality, frame)
			return img, quality_, ripeness_, size
		else:
			pass


	def create_models(self):
		#predict up to 3 items only
		self.model_mtl = Import_MTL("frozen_models/MTL_frozen_model.pb")
		self.model_fcnn = Import_Frcnn('frozen_models/frozen_inference_graph.pb')
		self.model_lr = load_graph('frozen_models/LR_frozen_model.pb')

	def terminate(self):
		self.stop.set()


def draw_boxes_scores(box_array, score_array, ripe_array, quality_array, frame):
	ripeness_dict = {0: 'Green', 1: 'Semi-Ripe', 2: 'Ripe'}
	quality_dict = {0: 'Good', 1: 'Defect'}
	cv2.rectangle(frame, (int(box_array[0]), int(box_array[2])), (int(box_array[1]), int(box_array[3])),(0,255,0),3)
	cv2.putText(frame, "Detection:  {0:.2f}".format(score_array), (10, 16), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,0))
	cv2.putText(frame, "Quality:  {}".format(quality_dict[int(np.argmax(quality_array, axis=1))]), (10, 32), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,0))
	cv2.putText(frame, "Ripeness:  {}".format(ripeness_dict[int(np.argmax(ripe_array, axis=1))]), (10, 48), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,0))
	return frame, quality_dict[int(np.argmax(quality_array, axis=1))], ripeness_dict[int(np.argmax(ripe_array, axis=1))]







class PhotoBoothApp:
	def __init__(self, vs):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.vs = vs
		self.frame = None
		self.thread = None
		self.stopEvent = None

		# initialize the root window and image panel
		self.root = tki.Tk()
		self.panel = None


		#self.arduinos = arduino("/dev/ttyUSB0", 9600)
		self.yo_thread = MyThread()


		w = tki.Label(self.root, text="Mango AI")
		w.pack()


		self.r = tki.Label(self.root, text="ripeness:")
		self.q = tki.Label(self.root, text="quality:")
		self.s = tki.Label(self.root, text="size:")
		self.r.pack()
		self.q.pack()
		self.s.pack()


		# create a button, that when pressed, will take the current
		# frame and save it to file
		btn = tki.Button(self.root, text="QUIT",
			command=self.onClose)
		btn.pack(side="bottom", fill="both", expand="yes", padx=10,
			pady=10)
		start_time = time.time()

		btn = tki.Button(self.root, text="process",
			command=self.run_process)
		timey  =(time.time() - start_time)

		btn.pack(side="bottom", fill="both", expand="yes", padx=10,
			pady=10)



		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop)
		self.thread.start()

	def videoLoop(self):
		# DISCLAIMER:
		# I'm not a GUI developer, nor do I even pretend to be. This
		# try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading


		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				# grab the frame from the video stream and resize it to
				# have a maximum width of 300 pixels
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=300)
		
				# OpenCV represents images in BGR order; however PIL
				# represents images in RGB order, so we need to swap
				# the channels, then convert to PIL and ImageTk format
		except RuntimeError:
			print("[INFO] caught a RuntimeError")


	def run_process(self):
		images  = None
		images, quality, ripeness, size = self.get_image()


		if quality[0] == quality[1]:
			qus = quality[0]
		else:
			qus = "Defect" 

		if ripeness[0] == ripeness[1]:
			ripss = ripeness[0]
		else:
			ripss = "Semi-Ripe"




		if images:
			if self.panel is None:
				self.panel = tki.Label(image=images[0])

				self.panel.pack(side="left", padx=10, pady=10)
				self.panel1 = tki.Label(image=images[1])
				self.panel1.pack(side="left", padx=10, pady=10)

			else:
				self.panel.configure(image=images[0])
				self.panel.image = images[0]
				self.panel1.configure(image=images[1])
				self.panel1.image = images[1]		
				self.r.configure(text = "Ripeness: {}".format(ripss))
				self.q.configure(text= "Quality: {}".format(qus))
				self.s.configure(text=size[0])


	def get_image(self):
		images = []
		quality_ = []
		ripeness_ = []
		sizes_ = []
		#grab 2 images, then process them
		results = self.process_image()
		images.append(results[0])
		quality_.append(results[1])
		ripeness_.append(results[2])
		sizes_.append(results[3])
		#self.arduinos.send_data(1)
		time.sleep(0.5)
		results = self.process_image()
		images.append(results[0])
		quality_.append(results[1])
		ripeness_.append(results[2])
		sizes_.append(results[3])

	
		return images, quality_, ripeness_, sizes_

	def process_image(self):
		image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
		try:
			image, quality, ripeness, size = self.yo_thread.run(image)
			image = Image.fromarray(image)
			image = ImageTk.PhotoImage(image)
			return image, quality, ripeness, size
		except:
			quality = "None"
			ripeness = "None"
			size = "None"
			image = Image.fromarray(image)
			image = ImageTk.PhotoImage(image)
			return image, quality, ripeness, size
			


	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.vs.stop()
		self.root.quit()



from imutils.video import VideoStream
import time
 

print("[INFO] warming up camera...")
vs = VideoStream(0).start()
time.sleep(2.0)
 
# start the app
pba = PhotoBoothApp(vs)
pba.root.mainloop()