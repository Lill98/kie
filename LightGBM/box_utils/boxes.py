from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

def points2xyxy(points):
    points = np.array(points)
    x1y1 = points.min(axis=0)
    x2y2 = points.max(axis=0)
    return np.concatenate((x1y1, x2y2), axis=0).astype(np.int32)


def xyxy2cxcy(bboxes):
    bboxes = np.array(bboxes).copy()
    cxcy = bboxes[[0, 1]] + (bboxes[[2, 3]]-bboxes[[0, 1]]) / 2
    wh = bboxes[[2, 3]] - bboxes[[0, 1]]
    return np.concatenate((cxcy ,wh), axis=0).astype(np.int32)


def xyxy2points(xyxy):
    return [[int(xyxy[0]), int(xyxy[1])], [int(xyxy[2]), int(xyxy[1])], [int(xyxy[2]), int(xyxy[3])], [int(xyxy[0]), int(xyxy[3])]]


def cal_degrees(a, b):
    v1 = [1, 0]
    v0 = [b[0] - a[0], b[1] - a[1]]
    deg = np.degrees(np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)))
    return deg


def boxes_distance(delta1, delta2):
    u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
    v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
    dist = np.linalg.norm(np.concatenate([u, v]))
    return dist


def dist_points(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist = np.linalg.norm(p1-p2)
    return dist


def normalize_scale_bbox(bbox, width, height):
    bbox = bbox.copy()
    bbox = np.array(bbox, dtype=np.float32)
    bbox[[0, 2]] /= width
    bbox[[1, 3]] /= height
    return bbox.tolist()


def unnormalize_scale_bbox(bbox, width, height):
    bbox = bbox.copy()
    bbox = np.array(bbox, dtype=np.float32)
    bbox[[0, 2]] *= width
    bbox[[1, 3]] *= height
    bbox = np.array(bbox, dtype=np.int32)
    return bbox.tolist()