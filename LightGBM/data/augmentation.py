from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np

class Augmentation(object):
    def __init__(self) -> None:
        pass
    
    def rotate_point(self,  origin, point, angle):
        ox, oy = origin
        px, py = point
    
        qx = ox + math.cos(angle) * (px-ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (py-oy) + math.cos(angle) * (py - oy)
        
        return int(qx), int(qy)
    
    def rotate_box(self, bbox, width, height):
        degrees = np.random.randint(-70, 70)
        points = [[int(bbox[0]), int(bbox[1])], [int(bbox[2]), int(bbox[1])], [int(bbox[2]), int(bbox[3])], [int(bbox[0]), int(bbox[3])]]
        cxcy = [(bbox[2]-bbox[0])/2, (bbox[3]-bbox[1])/2]
        rotated_points = [self.rotate_point(cxcy, pt, math.radians(degrees)) for pt in points]
        points = np.array(points)
        rotated_points = np.array(rotated_points)
        rotated_points[[0, 2]] = np.clip(rotated_points[[0, 2]], a_min=0.0, a_max=width)
        rotated_points[[1, 3]] = np.clip(rotated_points[[1, 3]], a_min=0.0, a_max=height)
        x_min, y_min = np.min(rotated_points, axis=0)
        x_max, y_max = np.max(rotated_points, axis=0)
        return [x_min, y_min, x_max, y_max]
    
    def translate_box(self, bbox, width, height):
        percent_rd = np.random.uniform(-0.2, 0.2)
        bbox = np.array(bbox, dtype=np.float32)
        bbox += (bbox * percent_rd)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], a_min=0.0, a_max=width)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], a_min=0.0, a_max=height)
        return bbox.tolist()
    
    def zoom_in_out_box(self, bbox, width, height):
        percent_rd = np.random.uniform(-0.2, 0.2)
        bbox = np.array(bbox, np.float32)
        bbox[[0, 2]] -= (bbox[[0, 2]] * percent_rd)
        bbox[[1, 3]] += (bbox[[1, 3]] * percent_rd)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], a_min=0.0, a_max=width)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], a_min=0.0, a_max=height)
        return bbox.tolist()
        
        
