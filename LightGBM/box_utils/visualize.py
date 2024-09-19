from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *
import os
import cv2

convert_labels = {
    'question': 'key',
    'answer': 'value',
    'header': 'title',
    'other': 'other',
    'o': 'other'
}

label2color = {
    "key": (0, 0, 255),
    "value": (255, 0, 0),
    "title": (0, 0, 255),
    "other": (255 ,255, 0),
    "o": (255, 255, 0),
    "linking": (150, 150, 0)
}

class Visualization:
    @classmethod
    def debug_ser(cls):
        with open(cfg.results_txt_path, 'r', encoding='utf-8') as f_res:
            results = f_res.readlines()
        for res_str in results:
            img_pth, res = res_str.strip().split('\t')
            res = eval(res)['ocr_info']
            cls.visualize_ser(img_pth, res, cfg.debug_path)

    def visualize_re(img_pth, labels):
        image = cv2.imread(img_pth)
        basename = os.path.basename(img_pth)
        for qa in labels:
            for obj in qa:
                if obj['label'] == 'question':
                    label_q = convert_labels[obj['label']]
                    bbox_q = points2xyxy(obj['points'])
                    image = cv2.rectangle(image, (bbox_q[0], bbox_q[1]), (
                        bbox_q[2], bbox_q[3]), color=label2color[label_q], thickness=2)
                    image = cv2.circle(image, ((
                        bbox_q[2]-bbox_q[0])/2, (bbox_q[3]-bbox_q[1])/2), radius=20, color=label2color[label_q], thickness=-1)
                    image = cv2.putText(image, str(label_q), (bbox_q[0], bbox_q[1]-10),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1, thickness=2, color=label2color[label_q],
                                        lineType=cv2.LINE_AA)
                for _obj in qa:
                    if _obj['label'] == 'answer':
                        label_a = convert_labels[_obj['label']]
                        bbox_a = points2xyxy(_obj['points'])
                        image = cv2.rectangle(image, (bbox_a[0], bbox_a[1]), (
                            bbox_a[2], bbox_a[3]), color=label2color[label_a], thickness=2)
                        image = cv2.circle(image, ((
                            bbox_a[2]-bbox_a[0])/2, (bbox_a[3]-bbox_a[1])/2), radius=20, color=label2color[label_a], thickness=-1)
                        image = cv2.line(image, ((bbox_q[2]-bbox_q[0])/2, (bbox_q[3]-bbox_q[1])/2), ((
                            bbox_a[2]-bbox_a[0])/2, (bbox_a[3]-bbox_a[1])/2), color=label2color['linking'], thickness=2)
                        image = cv2.putText(image, str(label_a), (bbox_a[0], bbox_a[1]-10),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=1, thickness=2, color=label2color[label_a],
                                            lineType=cv2.LINE_AA)

        os.makedirs(os.path.join(cfg.debug_path, 'ser'), exist_ok=True)
        cv2.imwrite(os.path.join(cfg.debug_path, 'ser', basename), image)

    @classmethod
    def visualize_ser(cls, img_pth, labels, save_dir):
        image = cv2.imread(img_pth)
        image_gt = image.copy()
        image_pred = image.copy()
        H, W, C = image.shape
        mask = np.ones((H, W * 2, C), dtype=image.dtype)
        basename = os.path.basename(img_pth)
        for obj in labels:
            image_gt = cls.draw_obj(image_gt, obj, 'label')
            image_pred = cls.draw_obj(image_pred, obj, 'pred')

        mask[0:H, 0:W] = image_gt
        mask[0:H, W:W*2] = image_pred
        
        os.makedirs(os.path.join(save_dir, 'ser_gt'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ser_pred'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ser_gt_pred'), exist_ok=True)
        
        cv2.imwrite(os.path.join(save_dir, 'ser_gt', basename), image_gt)
        cv2.imwrite(os.path.join(save_dir, 'ser_pred', basename), image_pred)
        cv2.imwrite(os.path.join(save_dir, 'ser_gt_pred', basename), mask)

    @classmethod
    def draw_obj(cls, image, obj, type_label='label'):
        label_q = convert_labels[obj[type_label].lower()]
        bbox_q = points2xyxy(obj['points'])
        image = cv2.rectangle(image, (bbox_q[0], bbox_q[1]), (
            bbox_q[2], bbox_q[3]), color=label2color[label_q], thickness=2)
        image = cv2.putText(image, str(label_q), (bbox_q[0], bbox_q[1]-10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, thickness=2, color=label2color[label_q],
                            lineType=cv2.LINE_AA)
        return image


if __name__ == "__main__":
    vis = Visualization.debug_ser()
