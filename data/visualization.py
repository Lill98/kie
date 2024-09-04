from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *
import os
import cv2
import pandas as pd

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
    def visualize_ser_re(cls, kv_data:pd.DataFrame, image):
        k_list = kv_data['k_id'].unique().tolist()
        image = image.astype(np.uint8)
        for k in k_list:
            kv_df = kv_data[(kv_data.k_id==k)&(kv_data.is_linking==1)]
            if kv_df.shape[0] == 0: continue
            k_box = kv_df.k_box.tolist()[0]
            image = cls.draw_obj(image, k_box, label='key', prob=None)
            image = cv2.circle(image, (int(k_box[2]-(k_box[2]-k_box[0])//2), int(k_box[3]-(k_box[3]-k_box[1])//2)), radius=10, color=label2color['key'], thickness=-1)
            v_boxes = kv_df.v_box.tolist()
            for v_box in v_boxes:
                image = cls.draw_obj(image, v_box, label='value', prob=None)
                image = cv2.line(image, (int(k_box[2]-(k_box[2]-k_box[0])//2), int(k_box[3]-(k_box[3]-k_box[1])//2)), 
                                (int(v_box[2]-(v_box[2]-v_box[0])//2), int(v_box[3]-(v_box[3]-v_box[1])//2)), color=label2color['linking'], thickness=2)
        return image
    
    @classmethod
    def draw_obj(cls, image, bbox, label, prob=None):
        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (
            int(bbox[2]), int(bbox[3])), color=label2color[label], thickness=2)
        if prob is not None:
            text = str(label) + f': {round(prob, 3)}'
        else:
            text = label
        image = cv2.putText(image, text , (int(bbox[0]), int(bbox[1])-10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, thickness=2, color=label2color[label],
                            lineType=cv2.LINE_AA)
        image = cv2.circle(image, (int(bbox[2]-(bbox[2]-bbox[0])//2), int(bbox[3]-(bbox[3]-bbox[1])//2)), radius=5, color=label2color['value'], thickness=-1)
        return image
    
    @classmethod
    def visualize_errors(cls, kv_df:pd.DataFrame, image:np.ndarray):
        kv_records = kv_df[['k_box', 'k_text', 'v_box', 'v_text']].to_records(index=False)
        for kv in kv_records:
            k_box, k_text, v_box, v_text = kv
            image = cls.draw_obj(image, k_box, 'key')
            image = cls.draw_obj(image, v_box, 'value')
            image = cv2.line(image, (int(k_box[2]-(k_box[2]-k_box[0])//2), int(k_box[3]-(k_box[3]-k_box[1])//2)), 
                                (int(v_box[2]-(v_box[2]-v_box[0])//2), int(v_box[3]-(v_box[3]-v_box[1])//2)), color=label2color['linking'], thickness=2)
        return image


if __name__ == "__main__":
    vis = Visualization.debug_ser()
