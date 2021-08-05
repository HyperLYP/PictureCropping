import tensorflow as tf
import cv2 
#读取图片 
img_data_tf = tf.io.read_file('datasets/7_3_1_1_input/1.jpg')
img_tf = tf.image.decode_jpeg(img_data_tf, channels=3)
#对原始数据解码，并裁剪指定去
decode_and_crop_tf = tf.image.decode_and_crop_jpeg(img_data_tf,[0,0,224,224])
crop_to_bb_tf = tf.image.crop_to_bounding_box(img_tf,224,224,224,224)
central_crop_tf = tf.image.central_crop(img_tf,0.5)

with tf.compat.v1.Session() as sess:
    
    img,decode_crop,crop_bb,central_crop = sess.run([img_tf,
                                                     decode_and_crop_tf,
                                                     crop_to_bb_tf,
                                                     central_crop_tf]) 
    #RGB转BGR
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) 
    decode_crop = cv2.cvtColor(decode_crop,cv2.COLOR_RGB2BGR)   
    crop_bb = cv2.cvtColor(crop_bb,cv2.COLOR_RGB2BGR)   
    central_crop = cv2.cvtColor(central_crop,cv2.COLOR_RGB2BGR) 
    #保存图片    
    cv2.imwrite('datasets/7_3_1_1_output/0_ori_img.jpg',img)
    cv2.imwrite('datasets/7_3_1_1_output/1_decode_crop.jpg',decode_crop)
    cv2.imwrite('datasets/7_3_1_1_output/2_crop_bb.jpg',crop_bb)
    cv2.imwrite('datasets/7_3_1_1_output/3_central_crop.jpg',central_crop)
