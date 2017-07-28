import os
import json


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


if __name__ == "__main__":
    preprocess(source_root_dir=r'Z:\Training\Recognition\Databases\Annotated',
               output_path=r'D:\GIT\deep-learning\object-detection\tensorflow-yolo\data\train_face.txt')