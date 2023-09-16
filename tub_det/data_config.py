import json
from pathlib import Path
import random
import cv2
import pandas as pd
from xml.etree import ElementTree as ET
import logging
import yaml
import numpy as np
import torch


def set_logging(log_file_path):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
        handlers=[logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()])


def set_random_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


class DataConfigManager:
    def __init__(self, data_root_path, config_file_path_dict):
        self.data_root_path = Path(data_root_path)
        self.config_file_path_dict = config_file_path_dict

    def generate(self):
        anno_info_list = self.parse_anno_info()
        self.generate_yolo_configs(anno_info_list)

    def parse_anno_info(self):
        anno_info_list = []
        anno_file_paths = list(self.data_root_path.rglob('*.xml'))

        for anno_file_path in anno_file_paths:
            xml_tree = ET.parse(anno_file_path.as_posix())
            root = xml_tree.getroot()

            filename = root.find('filename').text
            image_file_path = anno_file_path.parent / filename
            size = root.find('size')
            size_dict = self.parse_size(size)
            anno_info = {
                'image_file_path': image_file_path,
                'size': size_dict,
                'bnd_box_list': []
            }

            for object_iter in root.findall('object'):
                name = object_iter.find('name').text
                bnd_box = object_iter.find('bndbox')

                bnd_box_dict = self.parse_bnd_box(bnd_box)
                bnd_box_dict['class_name'] = name
                anno_info['bnd_box_list'].append(bnd_box_dict)

            anno_info_list.append(anno_info)

        return anno_info_list

    def generate_yolo_configs(self,
                              anno_info_list,
                              max_num_val_data=1000,
                              max_val_percent=0.2,
                              seed=7):
        config_file_path_dict = self.config_file_path_dict
        class_name_dict = {}

        for anno_info in anno_info_list:
            bnd_box_list = anno_info['bnd_box_list']

            for bnd_box_dict in bnd_box_list:
                class_name = bnd_box_dict['class_name']
                class_name_dict.setdefault(class_name, 0)
                class_name_dict[class_name] += 1

        class_names = list(class_name_dict.keys())

        for anno_info in anno_info_list:
            anno_contents = []

            size_dict = anno_info['size']
            image_width = size_dict['width']
            image_height = size_dict['height']

            bnd_box_list = anno_info['bnd_box_list']

            for bnd_box_dict in bnd_box_list:
                class_name = bnd_box_dict['class_name']
                class_index = class_names.index(class_name)

                x_min = bnd_box_dict['xmin']
                y_min = bnd_box_dict['ymin']
                x_max = bnd_box_dict['xmax']
                y_max = bnd_box_dict['ymax']

                normed_center_x = (x_min + x_max) / 2 / image_width
                normed_center_y = (y_min + y_max) / 2 / image_height
                normed_bbox_width = (x_max - x_min) / image_width
                normed_bbox_height = (y_max - y_min) / image_height
                line = '{} {} {} {} {}'.format(
                    class_index,
                    normed_center_x,
                    normed_center_y,
                    normed_bbox_width,
                    normed_bbox_height)
                anno_contents.append(line)

            image_file_path = Path(anno_info['image_file_path'])
            anno_config_file_path = image_file_path.with_suffix('.txt')

            with open(anno_config_file_path, 'w') as file_stream:
                for line in anno_contents:
                    file_stream.write('{}\n'.format(line))

        set_random_seed(seed)
        random.shuffle(anno_info_list)

        num_val_data = min(max_num_val_data,
                           int(len(anno_info_list) * max_val_percent))

        anno_infos_dict = {
        'train': anno_info_list[:-num_val_data],
        'val': anno_info_list[-num_val_data:]
        }

        for data_type, anno_infos in anno_infos_dict.items():
            message = r'{}: writing file {} with num_data {}'.format(
                data_type,
                config_file_path_dict[data_type],
                len(anno_infos))
            logging.info(message)

            with open(config_file_path_dict[data_type], 'w') as file_stream:
                for anno_info in anno_infos:
                    image_file_path = anno_info['image_file_path']
                    file_stream.write('{}\n'.format(image_file_path))

        dataset_config = {
            'path': self.data_root_path.as_posix(),
            'train': config_file_path_dict['train'].name,
            'val': config_file_path_dict['val'].name,
            'names': {
                class_index: class_name
                for class_index, class_name
                in enumerate(class_names)
            }
        }

        message = r'Writing dataset config file: {}'.format(
            config_file_path_dict['dataset'])
        logging.info(message)

        with open(config_file_path_dict['dataset'], 'w') as file_stream:
            yaml.dump(dataset_config, file_stream, indent=4)

    def parse_size(self, size):
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        size_dict = {
            'width': width,
            'height': height
        }

        return size_dict

    def parse_bnd_box(self, bnd_box):
        if not bnd_box:
            return None

        x_min = float(bnd_box.find('xmin').text)
        y_min = float(bnd_box.find('ymin').text)
        x_max = float(bnd_box.find('xmax').text)
        y_max = float(bnd_box.find('ymax').text)

        bnd_box_dict = {
            'xmin': x_min,
            'ymin': y_min,
            'xmax': x_max,
            'ymax': y_max
        }

        return bnd_box_dict


def main():
    data_root_path = Path(r'/home/data')

    config_file_path_dict = {
        'train': data_root_path / 'train.txt',
        'val': data_root_path / 'val.txt',
        'dataset': data_root_path / 'custom_dataset.yaml'
    }

    log_file_path = '/project/train/log/log.txt'

    set_logging(log_file_path)

    logging.info('=' * 80)
    logging.info('Start DataConfigManager')
    data_manager = DataConfigManager(data_root_path,
                                     config_file_path_dict)
    data_manager.generate()
    logging.info('End DataConfigManager')
    logging.info('=' * 80)

if __name__ == '__main__':
    main()
