# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
from paddle import nn
# from paddle import tensor, float32
# from paddle.tensor import 
from key_value_extractor.ppocr.losses.basic_loss import DMLLoss
# import torch

class VQASerTokenLayoutLMLoss(nn.Layer):
    def __init__(self, num_classes, key=None, weight_class=None):
        super().__init__()
        if weight_class is not None:
            # weight_class = tensor(weight_class,dtype=float32)
            weight_class = paddle.to_tensor(weight_class)
            print("create cross entropy with weight....,", weight_class)
            self.loss_class = nn.CrossEntropyLoss(weight=weight_class,reduction='mean')
        
        else:
            self.loss_class = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.ignore_index = self.loss_class.ignore_index
        self.key = key

    def forward(self, predicts, batch):
        if isinstance(predicts, dict) and self.key is not None:
            predicts = predicts[self.key]
        labels = batch[5]
        # print("-----labels", labels)
        attention_mask = batch[2]
        if attention_mask is not None:
            active_loss = attention_mask.reshape([-1, ]) == 1
            active_output = predicts.reshape(
                [-1, self.num_classes])[active_loss]
            active_label = labels.reshape([-1, ])[active_loss]
            loss = self.loss_class(active_output, active_label)
        else:
            loss = self.loss_class(
                predicts.reshape([-1, self.num_classes]),
                labels.reshape([-1, ]))
        return {'loss': loss}