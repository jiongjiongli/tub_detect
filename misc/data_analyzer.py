import json
import os
from pathlib import Path
import random
import re
from xml.etree import ElementTree as ET
import cv2
import html
import pandas as pd
import logging


def set_logging(log_file_path):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
        handlers=[logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()])


class DataAnalyzer:
    def __init__(self, data_root_path):
        self.data_root_path = Path(data_root_path)

    def parse(self):
        anno_file_paths = list(self.data_root_path.rglob('*.xml'))
        parse_results = []

        for anno_file_path in anno_file_paths:
            # print(anno_file_path.as_posix())

            xml_tree = ET.parse(anno_file_path.as_posix())

            root = xml_tree.getroot()

            self.verify_xml(root, anno_file_path)

            filename = root.find('filename').text
            image_file_path = anno_file_path.parent / filename

            if not re.match(r'^[a-zA-Z0-9_./]+$',
                            image_file_path.as_posix()):
                message = r'Error: {} image path {} invalid!'.format(
                    anno_file_path,
                    image_file_path)
                logging.error(message)

            # Refer to YOLOv8 method: img2label_paths
            for image_path_keyword in ['images']:
                keyword = r'{0}{1}{0}'.format(os.sep, image_path_keyword)
                if keyword in image_file_path.as_posix():
                    message = r'Error: {} image path {} contains invalid keyword {}!'.format(
                        anno_file_path,
                        image_file_path,
                        keyword)
                    logging.error(message)

            name_parts = image_file_path.name.split('.')
            if len(name_parts) - 1 != 1:
                message = r'Error: {} number of dot in image path {} {} != {}'.format(
                    anno_file_path,
                    image_file_path,
                    len(name_parts) - 1,
                    1)
                logging.error(message)

            if not re.match(r'^[a-zA-Z0-9_]+.[a-zA-Z]+$', image_file_path.name):
                message = r'Error: {} name of image path {} is invalid! {}'.format(
                    anno_file_path,
                    image_file_path,
                    image_file_path.name)
                logging.error(message)

            if not image_file_path.exists():
                message = r'Error: {} image {} not exist!'.format(
                anno_file_path,
                image_file_path)
                logging.error(message)
                continue

            size = root.find('size')
            size_dict = self.parse_size(size)

            img = cv2.imread(image_file_path.as_posix())
            image_shape = (size_dict['height'], size_dict['width'], 3)
            message = r'Error: {} image shape {} != {}'.format(
                image_file_path,
                img.shape,
                image_shape)

            if not (img.shape == image_shape):
                logging.error(message)

            obj_list = []

            for object_iter in root.findall('object'):
                name = object_iter.find('name').text
                bnd_box = object_iter.find('bndbox')
                attributes = object_iter.find('attributes')
                polygon = object_iter.find('polygon')

                truncated = object_iter.find('truncated')
                if truncated is not None:
                    is_truncated = bool(int(truncated.text))
                else:
                    is_truncated = False

                occluded = object_iter.find('occluded')
                if occluded is not None:
                    is_occluded = bool(int(occluded.text))
                else:
                    is_occluded = False

                difficult = object_iter.find('difficult')
                if difficult is not None:
                    is_difficult = bool(int(difficult.text))
                else:
                    is_difficult = False

                bnd_box_dict = self.parse_bnd_box(bnd_box)
                attribute_dict = self.parse_attributes(attributes)
                points = self.parse_polygon(polygon)

                obj = {
                    'name': name,
                    'is_truncated': is_truncated,
                    'is_occluded': is_occluded,
                    'is_difficult': is_difficult,
                    'bnd_box': bnd_box_dict,
                    'attribute': attribute_dict,
                    'points': points
                }

                obj_list.append(obj)

            parse_result = {
                'anno_file_path': anno_file_path.as_posix(),
                'image_file_path': image_file_path.as_posix(),
                'file_name': filename,
                'size': size_dict,
                'objects': obj_list
            }

            parse_results.append(parse_result)

        return parse_results

    def verify_xml(self, root, anno_file_path):
        child_info_dict = {
            'folder': 1,
            'filename': 1,
            'source': [0, 1],
            'size': 1,
            'segmented': [0, 1],
            'object': 0
        }

        self.verify_node(root,
                         child_info_dict,
                         anno_file_path)

        size = root.find('size')

        child_info_dict = {
            'width': 1,
            'height': 1,
            'depth': 1
        }

        self.verify_node(size,
                         child_info_dict,
                         anno_file_path)

        for object_iter in root.findall('object'):
            child_info_dict = {
                'name': 1,
                'truncated': 1,
                'occluded': 1,
                'difficult': 1,
                'attributes': [0, 1],
                'polygon': [0, 1],
                'bndbox': 1
            }

            self.verify_node(object_iter,
                             child_info_dict,
                             anno_file_path)

            bnd_box = object_iter.find('bndbox')

            if bnd_box:
                child_info_dict = {
                    'xmin': 1,
                    'ymin': 1,
                    'xmax': 1,
                    'ymax': 1
                }

                self.verify_node(bnd_box,
                                 child_info_dict,
                                 anno_file_path)

            attributes = object_iter.find('attributes')

            if attributes:
                child_info_dict = {
                    'attribute': 1
                }

                self.verify_node(attributes,
                                 child_info_dict,
                                 anno_file_path)

                for attribute in attributes.findall('attribute'):
                    child_info_dict = {
                        'name': 1,
                        'value': 1
                    }

                    self.verify_node(attribute,
                                     child_info_dict,
                                     anno_file_path)

            polygon = object_iter.find('polygon')

            if polygon:
                child_info_dict = {
                    'points': 1
                }

                self.verify_node(polygon,
                                 child_info_dict,
                                 anno_file_path)

    def verify_node(self, node, child_info_dict, anno_file_path):
        for child in node.findall('*'):
            message = r'Error: {} tag {} not in: {}'.format(
                anno_file_path, child.tag, child_info_dict)
            if not (child.tag in child_info_dict):
                logging.error(message)

        for key, count in child_info_dict.items():
            if isinstance(count, int):
                count = [count]

            if len(count) == 1 and count[0] == 0:
                continue

            children = node.findall(key)
            message = r'Error: {} tag {} count {} not in {}'.format(
                anno_file_path, key, len(children), count)

            if not (len(children) in count):
                logging.error(message)

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

    def parse_attributes(self, attributes):
        if not attributes:
            return None

        attribute_dict = {}

        for attribute in attributes.findall('attribute'):
            name = attribute.find('name').text
            value = attribute.find('value').text
            attribute_dict[name] = value

        if 'rotation' in attribute_dict:
            try:
                attribute_dict['rotation'] = float(attribute_dict['rotation'])
            except:
                message = r'Exception in parsing rotation: {}'.format(attribute_dict['rotation'])
                logging.exception(message)

        return attribute_dict

    def parse_polygon(self, polygon):
        if not polygon:
            return None

        points_text = polygon.find('points').text
        point_texts = points_text.split(';')
        points = []

        for point_text in point_texts:
            point_values = point_text.split(',')

            point = [float(point_value)
                for point_value in point_values]

            if not len(point) == 2:
                message = r'num_point_value: {} != 2'.format(point)
                logging.error(message)

            points.append(point)

        # assert len(points) in [4, 6], points
        return points

    def analyze(self, parse_results):
        self.print_numbers('num_gt_files', len(parse_results))
        num_image_file_dict = {}
        img_size_dict = {}
        num_gt_dict = {}
        num_class_name_dict = {}
        num_truncated_gt = 0
        num_occluded_gt = 0
        num_difficult_gt = 0
        num_attribute_dict = {}
        num_points_dict = {}

        for parse_result in parse_results:
            anno_file_path = Path(parse_result['anno_file_path'])
            image_file_path = Path(parse_result['image_file_path'])
            num_image_file_dict.setdefault(image_file_path.suffix, 0)
            num_image_file_dict[image_file_path.suffix] += 1

            size_dict = parse_result['size']
            img_size = (size_dict['width'], size_dict['height'])
            img_size_dict.setdefault(img_size, 0)
            img_size_dict[img_size] += 1

            obj_list = parse_result['objects']
            num_gt = len(obj_list)
            num_gt_dict.setdefault(num_gt, 0)
            num_gt_dict[num_gt] += 1

            for obj in obj_list:
                class_name = obj['name']
                num_class_name_dict.setdefault(class_name, 0)
                num_class_name_dict[class_name] += 1

                is_occluded = obj['is_occluded']
                if is_occluded:
                    num_occluded_gt += 1

                is_truncated = obj['is_truncated']
                if is_truncated:
                    num_truncated_gt += 1

                is_difficult = obj['is_difficult']
                if is_difficult:
                    num_difficult_gt += 1

                bnd_box_dict = obj['bnd_box']

                attribute_dict = obj['attribute']

                if attribute_dict:
                    for key, value in attribute_dict.items():
                        num_attribute_dict.setdefault(key, {})
                        num_attribute_dict[key].setdefault(value, 0)
                        num_attribute_dict[key][value] += 1

                points = obj['points']
                if points:
                    num_points = len(points)
                    num_points_dict.setdefault(num_points, 0)
                    num_points_dict[num_points] += 1

        self.print_numbers('num_image_file_dict', num_image_file_dict)
        self.print_numbers('img_size_dict', img_size_dict)
        self.print_numbers('num_gt_dict', num_gt_dict)
        self.print_numbers('num_class_name_dict', num_class_name_dict)
        self.print_numbers('num_occluded_gt', num_occluded_gt)
        self.print_numbers('num_truncated_gt', num_truncated_gt)
        self.print_numbers('num_difficult_gt', num_difficult_gt)
        self.print_numbers('num_attribute_dict', num_attribute_dict)
        self.print_numbers('num_points_dict', num_points_dict)

    def print_numbers(self, comment, numbers):
        logging.info('#' * 80)
        logging.info(comment)
        logging.info('-' * 80)
        logging.info(numbers)
        logging.info('#' * 80)

def main():
    log_file_path = '/project/train/log/log.txt'
    set_logging(log_file_path)

    logging.info('')
    logging.info('=' * 80)
    logging.info('Start DataAnalyzer')
    data_root_path = r'/home/data'
    data_analyzer = DataAnalyzer(data_root_path)
    parse_results = data_analyzer.parse()
    data_analyzer.analyze(parse_results)
    logging.info('End DataAnalyzer')
    logging.info('=' * 80)
    logging.info('')


if __name__ == '__main__':
    main()
