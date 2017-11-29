import tensorflow as tf
import cv2
import numpy as np
import collections
import time
from mango_parameters import *
from size_classifier import load_graph


import threading

configd = tf.ConfigProto()
configd.gpu_options.allow_growth=True


class Import_Frcnn():
	def __init__(self, location):
		self.graph_frcnn = tf.Graph()
		self.sess = tf.Session(graph=self.graph_frcnn, config=configd)
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
		self.sess = tf.Session(graph = self.graph_mtl, config=configd)
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
	if scores.item(0) > 0.8:
		ymin, xmin, ymax, xmax = boxes[0]
		box = [xmin * width, xmax * width, ymin * height, ymax * height]
		score = scores.item(0)
	
	else: 
		pass

	return box, score

	





class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.stop = threading.Event()
		self.create_models()

	def run(self, frame):
		# Load the VGG16 network

		self.predict(frame)
		

	def predict(self, frame):

		(boxes, scores, classes, num) = self.model_fcnn.run(frame)
		box_array, scores_ = get_box(boxes, scores, frame)
		if scores_:
			left, right, top, bottom  = box_array
			crop = frame[int(top):int(bottom), int(left):int(right)]
			quality, ripeness = self.model_mtl.run(crop)
			draw_boxes_scores(box_array, scores_, ripeness, quality)
		else:
			pass


	def create_models(self):
		#predict up to 3 items only
		self.model_mtl = Import_MTL("frozen_models/MTL_frozen_model.pb")
		self.model_fcnn = Import_Frcnn('frozen_models/frozen_inference_graph.pb')

	def terminate(self):
		self.stop.set()



def draw_boxes_scores(box_array, score_array, ripe_array, quality_array):
	ripeness_dict = {0: 'Green', 1: 'Semi-Ripe', 2: 'Ripe'}
	quality_dict = {0: 'Good', 1: 'Defect'}
	cv2.rectangle(frame, (int(box_array[0]), int(box_array[2])), (int(box_array[1]), int(box_array[3])),(0,255,0),3)
	cv2.putText(frame, "Detection:{0:.2f}".format(score_array), (int(box_array[0]),int(box_array[2]-6)), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,255,0))
	cv2.putText(frame, "Quality:{}".format(quality_dict[int(np.argmax(quality_array, axis=1))]), (int(box_array[0]),int(box_array[2]-17)), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,255,0))
	cv2.putText(frame, "Ripeness:{}".format(ripeness_dict[int(np.argmax(ripe_array, axis=1))]), (int(box_array[0]),int(box_array[2]-28)), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,255,0))




def main():
	global frame
	cap = cv2.VideoCapture(0)
	cap.set(3,1080)
	cap.set(4,720)
	if (cap.isOpened()):
		print("Camera OK")
	else:
		cap.open()

	yo_thread = MyThread()

	while(True):

		ret, frame = cap.read()
		frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		yo_thread.run(frame_)
		cv2.imshow("Classification", frame)
		cv2.namedWindow('Classification',cv2.WINDOW_NORMAL)

		if (cv2.waitKey(1) & 0xFF == ord('q')):
			break;

	cap.release()
	cv2.destroyAllWindows()
	yo_thread.terminate()
	sys.exit()



if __name__ == "__main__":
	main()




