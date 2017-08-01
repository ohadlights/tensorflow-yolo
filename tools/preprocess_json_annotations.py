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
                image_file_path = meta_file_path.replace('.meta', '.png')
                with open(meta_file_path) as f:
                    data = json.load(f)
                rect = data['rect']
                if rect is None:
                    continue
                output_file.write('{0} {1} {2} {3} {4} {5}\n'.format(image_file_path, rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3], 0))


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
            if height < 416 or width < 416 or channels != 3:
                print('Image size is invalid: {0}'.format(image_path))
                continue
            #l = adjust_bb_to_448(height, width, l, image_size)
            output_file.write(l)


def adjust_bb_to_448(height, width, sample, image_size):
    width_ratio = image_size / width
    height_ratio = image_size / height
    sample = sample.strip().split(' ')
    xmin = float(sample[1]) * width_ratio
    ymin = float(sample[2]) * height_ratio
    xmax = float(sample[3]) * width_ratio
    ymax = float(sample[4]) * height_ratio
    return '{0} {1} {2} {3} {4} {5}\n'.format(sample[0], int(xmin), int(ymin), int(xmax), int(ymax), 0)


if __name__ == "__main__":
    image_size = 416 #448
    # preprocess(source_root_dir=r'Z:\Training\Recognition\Databases\Annotated',
    #            output_path=r'D:\GIT\deep-learning\object-detection\tensorflow-yolo\data\train_face.txt')
    cut_down_train_list(list_path=r'D:\GIT\deep-learning\object-detection\tensorflow-yolo\data\train_face.txt',
                        output_path=r'D:\GIT\deep-learning\object-detection\tensorflow-yolo\data\train_face_reduced.txt',
                        cut_datasets=['morph', 'webface'],
                        keep_ratio=0.25,
                        image_size=image_size)