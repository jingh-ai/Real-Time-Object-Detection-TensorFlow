import xml.etree.ElementTree as ET
import os


def parse_cvat_keypoints_xml(xml_file):
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

    captionList = root.findall("track") 
    
    for caption in captionList:
        if caption.attrib['label'] == "person":
            group_id = int(caption.attrib["group_id"])
            for frame in caption:
                frame_id = int(frame.attrib['frame'])
                xtl = round(float(frame.attrib['xtl']))
                ytl = round(float(frame.attrib['ytl']))
                xbr = round(float(frame.attrib['xbr']))
                ybr = round(float(frame.attrib['ybr']))
                
                annotations[frame_id].append(
                    {
                        'frame_id': frame_id,
                        'id': group_id,
                        "bbox": [xtl,ytl,xbr,ybr],
                        'class': "person",
                        "keypoints": [0 for _ in range(51)]})
    
    for caption in captionList:
        if caption.attrib['label'] == "keypoints":
            group_id = int(caption.attrib["group_id"])
            points = caption.findall("track")
            for point in points:
                point_start_index = int(point.attrib["id"])*3
                point_list = point.findall("points")
                for frame in point_list:
                    frame_id = int(frame.attrib["frame"])
                    for anno in annotations[frame_id]:
                        if anno["id"] == group_id:
                            x,y = frame.attrib['points'].split(",")
                            if frame.attrib['outside'] == "1" or frame.attrib['occluded'] == "1":
                                v = 1
                            else:
                                v = 2
                            anno["keypoints"][point_start_index:point_start_index+3] = [float(x),float(y),v]

    return annotations


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