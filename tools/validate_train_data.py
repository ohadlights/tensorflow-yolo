import cv2


def validate_train_data(train_data_path, image_size):
    content = open(train_data_path).readlines()
    for sample in content:
        sample = sample.split(' ')
        image_path = sample[0]
        xmin = int(sample[1])
        ymin = int(sample[2])
        xmax = int(sample[3])
        ymax = int(sample[4])
        image = cv2.imread(image_path)
        #image = cv2.resize(image, (image_size, image_size))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imshow('debug', image)
        cv2.waitKey(30)


if __name__ == "__main__":
    image_size = 416 #448
    validate_train_data(train_data_path=r'D:\GIT\deep-learning\object-detection\tensorflow-yolo\data\train_face_reduced.txt',
                        image_size=image_size)