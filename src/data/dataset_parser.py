import xml.etree.ElementTree as ET
import os
import json
from src.data.data_type import Instance


def parse_cvat_instances_xml(xml_file):
    """
    Extract information from a CVAT-format XML file.

    Args:
        xml_file (str): The path for xml file

    Returns:
        Array with arrays of dictionaries. array[i][j] is a dictionary.
        the keys of each dictionary are:
        dict['frame_id']: integer value representing id of the frame.
        dict['id']: object id.
        dict['bbox']: Integer representing the x upper-left coordinate of the bounding box.
        dict['class']: String value representing object class.

    Raises:
        FileNotFoundError: If the xml file does not exist.
    """
    if not os.path.exists(xml_file):
        raise FileNotFoundError

    tree = ET.parse(xml_file)
    root = tree.getroot()
    task = root.find('meta').find('task')
    totalframe = int(task.find('size').text)
    annotations = [[] for _ in range(totalframe)]

    for obj in root.iter('track'):
        object_id = int(obj.attrib['id'])
        label = obj.attrib['label']
        for frame in obj:
            frame_id = int(frame.attrib['frame'])
            xtl = round(float(frame.attrib['xtl']))
            ytl = round(float(frame.attrib['ytl']))
            xbr = round(float(frame.attrib['xbr']))
            ybr = round(float(frame.attrib['ybr']))
            
            annotations[frame_id].append(
                    {
                        'frame_id': frame_id,
                        'id': object_id,
                        "bbox": [xtl,ytl,xbr,ybr],
                        'class': label
                    })
    return annotations


def parse_coco_instances(coco_dir):
    """
    Docstring for parse_coco_instances
    
    :param coco_dir: Description
    """
    annotations_file = os.path.join(coco_dir, 'annotations', 'instances_default.json')
    if not os.path.exists(annotations_file):
        raise FileNotFoundError

    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    images_info = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        
        bbox = ann['bbox']
        x, y, width, height = map(round, bbox)
        annotations_by_image[image_id].append({
            'id': ann['id'],
            'bbox': [x, y, x + width, y + height],
            'class': ann['category_id']
        })

    total_frames = len(images_info)
    parsed_annotations = [[] for _ in range(total_frames)]

    for image_id, anns in annotations_by_image.items():
        frame_index = images_info[image_id]['frame_index']
        parsed_annotations[frame_index] = anns

    return parsed_annotations

def parse_yolo_instances(yolo_dir, classes_file):
    """
    Docstring for parse_yolo_instances
    
    :param yolo_dir: Description
    :param classes_file: Description
    """
    if not os.path.exists(classes_file):
        raise FileNotFoundError

    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    annotations = []
    frame_files = sorted([f for f in os.listdir(yolo_dir) if f.endswith('.txt')])

    for frame_id, frame_file in enumerate(frame_files):
        frame_annotations = []
        with open(os.path.join(yolo_dir, frame_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert from YOLO format to bounding box format
                x1 = round((x_center - width / 2) * 1000)  # Assuming image width is 1000
                y1 = round((y_center - height / 2) * 1000) # Assuming image height is 1000
                x2 = round((x_center + width / 2) * 1000)
                y2 = round((y_center + height / 2) * 1000)

                frame_annotations.append({
                    'frame_id': frame_id,
                    'id': len(frame_annotations),
                    'bbox': [x1, y1, x2, y2],
                    'class': classes[class_id]
                })
        annotations.append(frame_annotations)

    return annotations
