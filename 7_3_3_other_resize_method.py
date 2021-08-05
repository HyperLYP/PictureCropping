import tensorflow as tf
import cv2 
#读取图片   
img_data_tf = tf.read_file('datasets/imgs/test.jpg')
img_tf = tf.image.decode_jpeg(img_data_tf, channels=3) 
#在索引0处添加一个新的维度，即将[height,width,channels]
#变为[1,height,width,channels]
img_tf = tf.expand_dims(img_tf,axis=0)
#分别执行不同的Resize算法
area_tf = tf.image.resize_area(img_tf,(128,64))
bicubic_tf =tf.image.resize_bicubic(img_tf,(64,128))
bilinear_tf = tf.image.resize_bilinear(img_tf,(128,128))
nearest_neighbor_tf = tf.image.resize_nearest_neighbor(img_tf,(128,256))
#去掉第一个维度
area_tf = tf.squeeze(area_tf,axis=0)
bicubic_tf = tf.squeeze(bicubic_tf,axis=0)
bilinear_tf = tf.squeeze(bilinear_tf,axis=0)
nearest_neighbor_tf = tf.squeeze(nearest_neighbor_tf,axis=0)
with tf.Session() as sess: 
    area,bicubic,bilinear,nearest_neighbor = sess.run([area_tf,
                                                       bicubic_tf,
                                                       bilinear_tf,
                                                       nearest_neighbor_tf])     
    
    area = cv2.cvtColor(area,cv2.COLOR_RGB2BGR) 
    bicubic = cv2.cvtColor(bicubic,cv2.COLOR_RGB2BGR) 
    bilinear = cv2.cvtColor(bilinear,cv2.COLOR_RGB2BGR) 
    nearest_neighbor = cv2.cvtColor(nearest_neighbor,cv2.COLOR_RGB2BGR) 
    cv2.imwrite('datasets/7_3_3_outputs/area.jpg',area) 
    cv2.imwrite('datasets/7_3_3_outputs/bicubic.jpg',bicubic) 
    cv2.imwrite('datasets/7_3_3_outputs/bilinear.jpg',bilinear) 
    cv2.imwrite('datasets/7_3_3_outputs/nearest_neighbor.jpg',nearest_neighbor)
