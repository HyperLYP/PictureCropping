import tensorflow as tf
import cv2 
#读取图片   
img_data_tf = tf.read_file('datasets/imgs/test.jpg')
img_tf = tf.image.decode_jpeg(img_data_tf, channels=3) 
 
#原始图像的宽高分别为256和256
#将原始图像Resize到[height,width]=[256,128]
img_1_tf = tf.image.resize_image_with_crop_or_pad(img_tf,
                                                  target_height=256,
                                                  target_width=128) 
#将原始图像Resize到[height,width]=[128,384]
img_2_tf = tf.image.resize_image_with_crop_or_pad(img_tf,
                                                  target_height=128,
                                                  target_width=384) 

with tf.Session() as sess: 
    img_1,img_2 = sess.run([img_1_tf,img_2_tf])   
    img_1 = cv2.cvtColor(img_1,cv2.COLOR_RGB2BGR)   
    img_2 = cv2.cvtColor(img_2,cv2.COLOR_RGB2BGR)  
    cv2.imwrite('datasets/7_3_2_2_outputs/crop_pad_1.jpg',img_1) 
    cv2.imwrite('datasets/7_3_2_2_outputs/crop_pad_2.jpg',img_2)
