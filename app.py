# -*- coding:utf-8 -*-  
import tensorflow as tf
import numpy as np
from PIL import Image
import test
import matplotlib.pyplot as plt

def pre_pic(picName):
# 将源图片转化为适合喂入神经网络的[1,784]格式
	img = Image.open(picName)
	reIm = img.resize((28,28),Image.ANTIALIAS)
	im_arr = np.array(reIm.convert('L'))
	threshold = 30
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 256-im_arr[i][j]
			if im_arr[i][j] < threshold:
				im_arr[i][j] = 1
			else:
				im_arr[i][j] = 200				
	plt.imshow(im_arr)
	plt.show()
	nm_arr = im_arr.reshape([1,784])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr,(1.0/255.0))
	
	return img_ready 

def restore_model(testPicArr):
# 识别图片
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32,[None,784])
		y = test.forward(x,1.0)
		preValue = tf.argmax(y,1)
		saver = tf.train.Saver()
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(test.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
				preValue = sess.run(preValue,feed_dict = {x:testPicArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1

def application():
	testFlag = True
	#while testFlag:
	#	testPic = raw_input("Please input the address of your picture: ")
	with tf.Session() as sess:
		for i in range(10):
			testPic = 'numPic/'+ str(i)+'.png'
			print testPic,i
			testPicArr = pre_pic(testPic)
			preValue = restore_model(testPicArr)
			if preValue > -1:
				print("The prediction number is", preValue)
if __name__ == "__main__":
	application()
