import os
import json
import random
import cv2


def preprocess(source_root_dir, output_path):
    with open(output_path, 'w') as output_file:
        for root, dirs, files in os.walk(source_root_dir):
            meta_files = filter(lambda x: x.endswith('.meta'), files)
            for meta_file in meta_files:
                meta_file_path = os.path.join(root, meta_file)
                #image_file_path = meta_file_path.replace('.meta', '.png')
                with open(meta_file_path) as f:
                    data = json.load(f)
                if 'image_path' in data:
                    image_file_path = data['image_path']
                    if 'rect' in data and 'dlib_rect' in data: # checking dlib rectangle to get some assurance that the face is somewhat frontal
                        rect = data['rect']
                        landmarks = data['landmarks']
                        landmarks = [landmarks[14][0], landmarks[14][1],
                                     landmarks[22][0], landmarks[22][1],
                                     landmarks[29][0], landmarks[29][1],
                                     landmarks[33][0], landmarks[33][1],
                                     landmarks[39][0], landmarks[39][1]]
                        output_file.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(image_file_path, rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3], 0, ' '.join(str(x) for x in landmarks)))
                else:
                    print('image path missing from meta file: {}'.format(meta_file_path))


def cut_down_train_list(list_path, output_path, cut_datasets, keep_ratio, image_size):
    content = open(list_path).readlines()
    chance = int(1/keep_ratio)
    with open(output_path, 'w') as output_file:
        for l in content:
            image_path = l.split(' ')[0]
            print(image_path)
            dataset = image_path.split('\\')[-3]
            if dataset in cut_datasets:
                rand = random.randint(1, chance)
                if rand != 1:
                    continue
            image = cv2.imread(image_path)
            if image is None:
                print('Image is none: {0}'.format(image_path))
                continue
            height, width, channels = image.shape
            if height < image_size or width < image_size or channels != 3:
                print('Image size is invalid: {0}'.format(image_path))
                continue
            output_file.write(l)


if __name__ == "__main__":
    preprocess(source_root_dir=r'Z:\Training\Detection\Annotations',
               output_path=r'D:\GIT\deep-learning\object-detection\tensorflow-yolo\data\train_face.txt')
    # cut_down_train_list(list_path=r'D:\GIT\deep-learning\object-detection\tensorflow-yolo\data\train_face.txt',
    #                     output_path=r'D:\GIT\deep-learning\object-detection\tensorflow-yolo\data\train_face_reduced.txt',
    #                     cut_datasets=['morph', 'webface'],
    #                     keep_ratio=0.25,
    #                     image_size=416)
