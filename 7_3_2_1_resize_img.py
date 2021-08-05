import tensorflow as tf
import cv2 
#读取图片  
img_path_tf=tf.convert_to_tensor(['datasets/imgs/test.jpg'],dtype=tf.string)
[img_path_queue] = tf.train.slice_input_producer([img_path_tf])
img_data_tf = tf.read_file(img_path_queue)
img_tf = tf.image.decode_jpeg(img_data_tf, channels=3)
img_tf.set_shape((256,256,3))
img_batch=tf.train.batch([img_tf],batch_size=4)
boxes=[[0.0,0.0,0.5,0.5],
       [0.0,0.5,0.5,1.0],
       [0.5,0.0,1.0,0.5],
       [0.5,0.5,1.0,1.0],
      ] 
#对图像裁剪并Resize
crop_and_resize_tf = tf.image.crop_and_resize(img_batch,boxes,[0,1,2,3],[256,256]) 

with tf.Session() as sess:
    
    coord = tf.train.Coordinator()
    #启动所有队列线程，使得每个队列源源不断执行入队和出队
    threads = tf.train.start_queue_runners(coord=coord)
    crop_and_resize = sess.run(crop_and_resize_tf)  
    for i in range(4):
        img = crop_and_resize[i]
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)   
        cv2.imwrite('datasets/7_3_2_1_outputs/crop_resize_%d.jpg'%i,img) 
     
    coord.request_stop()
    coord.join(threads)
