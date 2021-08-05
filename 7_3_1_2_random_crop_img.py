import tensorflow as tf
import cv2 
#读取图片 
img_path_tf=tf.convert_to_tensor(['datasets/imgs/test.jpg'],dtype=tf.string)
[img_path_queue] = tf.train.slice_input_producer([img_path_tf])
img_data_tf = tf.read_file(img_path_queue)
#img_tf的Shape为[256,256,3]  
img_tf = tf.image.decode_jpeg(img_data_tf, channels=3)
#对图像随机裁剪
random_crop_tf = tf.random_crop(img_tf,[160,160,3]) 

with tf.Session() as sess:
    
    coord = tf.train.Coordinator()
    #启动所有队列线程，使得每个队列源源不断执行入队和出队
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(5):
        random_crop = sess.run(random_crop_tf) 
        random_crop = cv2.cvtColor(random_crop,cv2.COLOR_RGB2BGR)   
        cv2.imwrite('datasets/7_3_1_2_outputs/random_%d.jpg'%i,random_crop) 
     
    coord.request_stop()
    coord.join(threads)
