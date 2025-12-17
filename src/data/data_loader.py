from typing import Generator
from train.model_train.utils.xml_parser import parse_voc_xml
from train.model_train.models.model import preprocess_true_boxes
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2 as cv


class DataLoader(object):
    def __init__(self, dataset_directory, class_names, anchor_boxes, batch_size, fix_input_size, input_height=1920, input_width=1056):
        self.dataset_dir = dataset_directory
        self.class_names = class_names
        self.batch_size = batch_size
        self.fix_input_size = fix_input_size
        self.input_shape = [input_height, input_width]
        self.anchors = anchor_boxes
        self.train_data_size = 0
        self.val_data_size = 0

        self.annotation_lines = self.dataset_converter(self.dataset_dir, self.class_names)

    def convert_annotation(self, dataset_path, image_id, class_names):
        annotation_line = '%s/Images/%s' % (dataset_path, image_id)
        annotation = parse_voc_xml('%s/Annotations/%s.xml' % (dataset_path, image_id))
        for obj in annotation:
            cls = obj['class']
            if cls not in class_names:
                continue
            cls_id = class_names.index(cls)
            bbox = (int(obj['xtl']), int(obj['ytl']), int(obj['xbr']), int(obj['ybr']))
            annotation_line += " " + ",".join([str(a) for a in bbox]) + ',' + str(cls_id)
        return annotation_line

    def dataset_converter(self, dataset_path, class_names):
        annotation_lines = []
        for image_set in ['trainval']:
            with open('%s/ImageSets/%s.txt' % (dataset_path, image_set)) as f:
                image_ids = f.read().splitlines()
            for image_id in image_ids:
                annotation_line = self.convert_annotation(dataset_path, image_id, class_names)
                annotation_lines.append(annotation_line)
        return annotation_lines

    def data_generator(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        vertical_images = []
        horizontal_images = []

        # separate vertical and horizontal images into two lists
        for annot_line in annotation_lines:
            camera_art = annot_line.split()[0].split('/')[-1][0]
            # ptz -> horizontal  ;  ax -> vertical
            if camera_art == "p":
                horizontal_images.append(annot_line)
            else:
                vertical_images.append(annot_line)

        ver_images_tmp = vertical_images.copy()
        hor_images_tmp = horizontal_images.copy()

        while True:
            image_data = []
            box_data = []
            
            # select a horizontal or vertical images for a batch
            if len(ver_images_tmp) > batch_size and len(hor_images_tmp) > batch_size:
                if np.random.rand() > 0.5:
                    images_to_used = hor_images_tmp
                else:
                    images_to_used = ver_images_tmp
            else:
                # make sure that all images will be used in a epoch
                if len(ver_images_tmp) > batch_size:
                    images_to_used = ver_images_tmp
                elif len(hor_images_tmp) > batch_size:
                    images_to_used = hor_images_tmp
                else:
                    ver_images_tmp = vertical_images.copy()
                    hor_images_tmp = horizontal_images.copy()
                    np.random.shuffle( ver_images_tmp )
                    np.random.shuffle( hor_images_tmp )
                    continue

            while len(image_data) < batch_size:
                annot_line = images_to_used.pop()
                image, box, input_shape = self.get_random_data(annot_line, input_shape, random=True)
                image_data.append(image)
                box_data.append(box)

            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)

    def data_generator_wrapper(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0: return None
        return self.data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

    def get_random_data(self,
            annotation_line,
            input_shape,
            random=True,
            max_boxes=50,
            jitter=.3,
            hue=.1,
            sat=1.5,
            val=1.5,
            proc_img=True):
        '''random preprocessing for real-time data augmentation'''
        line = annotation_line.split()
        try:
            image = Image.open(line[0] + '.png')
        except:
            image = Image.open(line[0] + '.jpg')
        iw, ih = image.size

        if self.fix_input_size:
            h, w = input_shape
        else:
            if image.size != (None, None) and iw % 32 == 0 and ih % 32 == 0:
                w = iw
                h = ih
            else:
                h = ih - ih % 32
                w = iw - iw % 32

        box = np.array([np.array(list(map(int, box.split(','))))
                        for box in line[1:]])

        if not random:
            # resize image
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image_data = 0
            if proc_img:
                image = image.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('RGB', (w, h), (128, 128, 128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image) / 255.

            # correct boxes
            box_data = np.zeros((max_boxes, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                if len(box) > max_boxes:
                    box = box[:max_boxes]
                box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
                box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
                box_data[:len(box)] = box

            return image_data, box_data, [h, w]

        # resize image
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / \
                 self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data, [h, w]

    def image_preporcess(self, image, target_size, gt_boxes=None):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded
        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_data_generator(self) -> (Generator, Generator):
        lines_trainval = self.annotation_lines
        np.random.seed(10101)
        np.random.shuffle(lines_trainval)
        np.random.seed(None)
        self.train_data_size = int(len(lines_trainval) * 0.8)
        lines_train = lines_trainval[:self.train_data_size]
        lines_val = lines_trainval[self.train_data_size:]
        self.val_data_size = len(lines_val)
        return self.data_generator_wrapper(lines_train, self.batch_size, self.input_shape, self.anchors,len(self.class_names)), \
               self.data_generator_wrapper(lines_val, self.batch_size, self.input_shape, self.anchors,len(self.class_names))

    def get_train_data_size(self) -> int:
        return self.train_data_size

    def get_validation_data_size(self) -> int:
        return self.val_data_size


