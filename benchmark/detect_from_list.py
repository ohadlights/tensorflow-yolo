import time
from yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf
import cv2
import numpy as np


def process_predicts(predicts, image_size):
  p_classes = predicts[0, :, :, 0:1]
  C = predicts[0, :, :, 1:3]
  coordinate = predicts[0, :, :, 3:]

  p_classes = np.reshape(p_classes, (13, 13, 1, 1))
  C = np.reshape(C, (13, 13, 2, 1))

  P = C * p_classes

  #print P[5,1, 0, :]

  index = np.argmax(P)

  index = np.unravel_index(index, P.shape)

  class_num = index[3]

  coordinate = np.reshape(coordinate, (13, 13, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (image_size/13.0)
  ycenter = (index[0] + ycenter) * (image_size/13.0)

  w = w * image_size
  h = h * image_size

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num


def adjust_bb(height, width, xmin, ymin, xmax, ymax, image_size):
    width_ratio = width / image_size
    height_ratio = height / image_size
    xmin *= width_ratio
    xmax *= width_ratio
    ymin *= height_ratio
    ymax *= height_ratio
    return xmin, ymin, xmax, ymax


def detect_face(sess, image_placeholder, predicts, image_path, image_size):
    np_img = cv2.imread(image_path)
    height, width, channels = np_img.shape
    resized_img = cv2.resize(np_img, (image_size, image_size))
    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    np_img = np_img.astype(np.float32)

    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, image_size, image_size, 3))

    predict_start_time = time.time()

    np_predict = sess.run(predicts, feed_dict={image_placeholder: np_img})

    xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict, image_size)

    predict_time = time.time() - predict_start_time

    # cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
    # cv2.imshow('debug', resized_img)
    # cv2.waitKey(0)

    xmin, ymin, xmax, ymax = adjust_bb(height, width, xmin, ymin, xmax, ymax, image_size)

    return xmin, ymin, xmax, ymax, predict_time


def detect(input_list_path, output_list_path, image_size):
    common_params = {'image_size': image_size, 'num_classes': 1,
                     'batch_size': 1}
    net_params = {'cell_size': 13, 'boxes_per_cell': 2, 'weight_decay': 0.0005}

    net = YoloTinyNet(common_params, net_params, test=True)

    image = tf.placeholder(tf.float32, (1, image_size, image_size, 3))
    predicts, landmarks_predicts = net.inference(image)

    with tf.Session() as sess:

        saver = tf.train.Saver(net.trainable_collection)

        saver.restore(sess, r'..\models\train_face\model.ckpt-10000')

        image_list = open(input_list_path).readlines()

        total_predict_time = 0

        with open(output_list_path, 'w') as output_file:
            for i in range(0, len(image_list)):
                image_path = image_list[i].split(' ')[0]
                xmin, ymin, xmax, ymax, predict_time = detect_face(sess=sess, image_placeholder=image, predicts=predicts, image_path=image_path, image_size=image_size)

                total_predict_time += predict_time

                output_file.write('{} {} {} {}\n'.format(xmin, ymin, xmax, ymax))

                if i % 100 == 0:
                    np_img = cv2.imread(image_path)
                    resized_img = np_img#cv2.resize(np_img, (image_size, image_size))
                    cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
                    cv2.imshow('debug', resized_img)
                    cv2.waitKey(1)
                    print('predict time = {}'.format(predict_time))

        average_predict_time = total_predict_time / len(image_list)
        print('average_predict_time = {}'.format(average_predict_time))


if __name__ == "__main__":
    detect(input_list_path=r'D:\FaceWorkspace\detection\benchmarks\images.txt',
           output_list_path=r'D:\FaceWorkspace\detection\benchmarks\detections_temp.txt',
           image_size=416)



