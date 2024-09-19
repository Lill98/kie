# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import paddle
import paddle.distributed as dist

from key_value_extractor.ppocr.data import create_operators, transform
from key_value_extractor.ppocr.modeling.architectures import build_model
from key_value_extractor.ppocr.postprocess import build_post_process
from key_value_extractor.ppocr.utils.save_load import load_model
from key_value_extractor.ppocr.utils.visual import draw_re_results
from key_value_extractor.ppocr.utils.logging import get_logger
from key_value_extractor.ppocr.utils.utility import get_image_file_list, load_vqa_bio_label_maps, print_dict
from key_value_extractor.tools.program import ArgsParser, load_config, merge_config
from key_value_extractor.tools.infer_kie_token_ser import SerPredictor


class ReArgsParser(ArgsParser):
    def __init__(self):
        super(ReArgsParser, self).__init__()
        self.add_argument(
            "-c_ser", "--config_ser", help="ser configuration file to use")
        self.add_argument(
            "-o_ser",
            "--opt_ser",
            nargs='+',
            help="set ser configuration options ")

    def parse_args(self, argv=None):
        args = super(ReArgsParser, self).parse_args(argv)
        assert args.config_ser is not None, \
            "Please specify --config_ser=ser_configure_file_path."
        args.opt_ser = self._parse_opt(args.opt_ser)
        return args


def make_input(ser_inputs, ser_results):
    entities_labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2}
    batch_size, max_seq_len = ser_inputs[0].shape[:2]
    entities = ser_inputs[8][0]
    ser_results = ser_results[0]
    assert len(entities) == len(ser_results)

    # entities
    start = []
    end = []
    label = []
    entity_idx_dict = {}
    for i, (res, entity) in enumerate(zip(ser_results, entities)):
        if res['pred'] == 'O':
            continue
        entity_idx_dict[len(start)] = i
        start.append(entity['start'])
        end.append(entity['end'])
        label.append(entities_labels[res['pred']])

    entities = np.full([max_seq_len + 1, 3], fill_value=-1, dtype=np.int64)
    entities[0, 0] = len(start)
    entities[1:len(start) + 1, 0] = start
    entities[0, 1] = len(end)
    entities[1:len(end) + 1, 1] = end
    entities[0, 2] = len(label)
    entities[1:len(label) + 1, 2] = label

    # relations
    head = []
    tail = []
    for i in range(len(label)):
        for j in range(len(label)):
            if label[i] == 1 and label[j] == 2:
                head.append(i)
                tail.append(j)

    relations = np.full([len(head) + 1, 2], fill_value=-1, dtype=np.int64)
    relations[0, 0] = len(head)
    relations[1:len(head) + 1, 0] = head
    relations[0, 1] = len(tail)
    relations[1:len(tail) + 1, 1] = tail

    entities = np.expand_dims(entities, axis=0)
    entities = np.repeat(entities, batch_size, axis=0)
    relations = np.expand_dims(relations, axis=0)
    relations = np.repeat(relations, batch_size, axis=0)

    # remove ocr_info segment_offset_id and label in ser input
    if isinstance(ser_inputs[0], paddle.Tensor):
        entities = paddle.to_tensor(entities)
        relations = paddle.to_tensor(relations)
    ser_inputs = ser_inputs[:5] + [entities, relations]

    entity_idx_dict_batch = []
    for b in range(batch_size):
        entity_idx_dict_batch.append(entity_idx_dict)
    return ser_inputs, entity_idx_dict_batch


class RePredictor(object):
    def __init__(self, config):
        self.use_label = False
        global_config = config['Global']
        
        # build post process
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)
        # build model
        self.model = build_model(config['Architecture'])
        load_model(
            config, self.model, model_type=config['Architecture']["model_type"])
        self.model.eval()

    def __call__(self, ser_results, ser_inputs):
        # ser_results, ser_inputs = self.ser_engine(data)
        if self.use_label:
            print("infer with label.............")
            pred_id_mapping = {"other": 0,
                               "key": 1, "value": 3, "title": 5}
            pred_mapping = {"other": "O", "key": "KEY",
                            "value": "VALUE", "title": "TITLE"}

            for i in ser_results[0]:
                i["pred_id"] = pred_id_mapping[i["label"]]
                i["pred"] = pred_mapping[i["label"]]

            for i in ser_inputs[-2][0]:
                i["pred_id"] = pred_id_mapping[i["label"]]
                i["pred"] = pred_mapping[i["label"]]
                
        re_input, entity_idx_dict_batch = make_input(ser_inputs, ser_results)

        if self.model.backbone.use_visual_backbone is False:
            re_input.pop(4)
        preds = self.model(re_input)
        post_result = self.post_process_class(
            preds,
            ser_results=ser_results,
            entity_idx_dict_batch=entity_idx_dict_batch)
        # print(post_result)
        return post_result

def preprocess():
    FLAGS = ReArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)

    ser_config = load_config(FLAGS.config_ser)
    ser_config = merge_config(ser_config, FLAGS.opt_ser)

    logger = get_logger()

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']

    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id) if use_gpu else 'cpu'
    device = paddle.set_device(device)

    logger.info('{} re config {}'.format('*' * 10, '*' * 10))
    print_dict(config, logger)
    logger.info('\n')
    logger.info('{} ser config {}'.format('*' * 10, '*' * 10))
    print_dict(ser_config, logger)
    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    return config, ser_config, device, logger

