import os
import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf
import cv2
import numpy as np

# classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
classes_name = ["face"]


def process_landmarks_predicts(predicts, index):
    predicts = predicts[0, index[0], index[1], index[2], :]
    landmarks = []
    for i in range(0, 10, 2):
        x = int((predicts[i] + index[1]) * (416 / 13.0))
        y = int((predicts[i+1] + index[0]) * (416 / 13.0))
        landmarks += [[x, y]]
    return landmarks


def process_predicts(predicts):
    p_classes = predicts[0, :, :, 0:1]
    C = predicts[0, :, :, 1:3]
    coordinate = predicts[0, :, :, 3:]

    p_classes = np.reshape(p_classes, (13, 13, 1, 1))
    C = np.reshape(C, (13, 13, 2, 1))

    P = C * p_classes

    # print P[5,1, 0, :]

    index = np.argmax(P)

    index = np.unravel_index(index, P.shape)

    class_num = index[3]

    coordinate = np.reshape(coordinate, (13, 13, 2, 4))

    max_coordinate = coordinate[index[0], index[1], index[2], :]

    xcenter = max_coordinate[0]
    ycenter = max_coordinate[1]
    w = max_coordinate[2]
    h = max_coordinate[3]

    xcenter = (index[1] + xcenter) * (416 / 13.0)
    ycenter = (index[0] + ycenter) * (416 / 13.0)

    w = w * 416
    h = h * 416

    xmin = xcenter - w / 2.0
    ymin = ycenter - h / 2.0

    xmax = xmin + w
    ymax = ymin + h

    return xmin, ymin, xmax, ymax, class_num, index


common_params = {'image_size': 416, 'num_classes': 1,
                 'batch_size': 1}
net_params = {'cell_size': 13, 'boxes_per_cell': 2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 416, 416, 3))
bb_predicts, landmarks_predict = net.inference(image)

with tf.Session() as sess:

    saver = tf.train.Saver(net.trainable_collection)
    saver.restore(sess, 'models/train_face_new - Copy (2)/model.ckpt-4000')

    for image_file in os.listdir('images'):

        np_img = cv2.imread('images\\' + image_file)
        resized_img = cv2.resize(np_img, (416, 416))
        np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        np_img = np_img.astype(np.float32)

        np_img = np_img / 255.0 * 2 - 1
        np_img = np.reshape(np_img, (1, 416, 416, 3))

        np_predict, np_landmarks_predict = sess.run([bb_predicts, landmarks_predict], feed_dict={image: np_img})

        xmin, ymin, xmax, ymax, class_num, index = process_predicts(np_predict)
        landmarks = process_landmarks_predicts(np_landmarks_predict, index)
        class_name = classes_name[class_num]
        cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        for i in range(len(landmarks)):
            cv2.circle(resized_img, (landmarks[i][0], landmarks[i][1]), 1, (0, 255, 0), 2)
        #cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
        cv2.imshow('result', resized_img)
        cv2.waitKey(0)
        #cv2.imwrite('001_01_01_050_00_out.jpg', resized_img)

