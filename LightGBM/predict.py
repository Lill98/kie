from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from box_utils.boxes import *
from data.visualization import Visualization

import os
import cv2
import glob
import pickle
import itertools
import pandas as pd
from tqdm import tqdm
from modules.kv_embedding import KVEmbedding
import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score

class Predictor(object):
    """Inference class LightGBM model


    """
    def __init__(self, lightgbm_config) -> None:
        self.lightgbm_config = lightgbm_config
        self.cols = ['k_id', 'k_text', 'k_box', 'v_id', 'v_text', 'v_box', 'k_embed', 'v_embed', 'width', 'height', 'fname']
        self.scaler_path = os.path.join(self.lightgbm_config["scaler_path"], 'scaler.pkl')
        self.model_path = os.path.join(self.lightgbm_config["model_path"], 'clf.pkl')
        self.model_classifier = self.load_model()
        self.dirname = self.lightgbm_config["image_path"]
        self.scaler = self.load_scaler()
        self.device = lightgbm_config["device"]
        self.kv_embed = KVEmbedding(self.device)
    
    def load_model(self):
        """Load the model

   
        """
        print(f'=====================Loading model post processing relation from {self.model_path}')
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f_model:
                model = pickle.load(f_model)
            f_model.close()
            return model
        else:
            print("Path to model not exist !", self.model_path)
            
    def load_label_data(self, type_data='val'):
        data_pth = self.lightgbm_config["label_path"]
        labels = []
        for pth in glob.glob(os.path.join(data_pth, type_data, "*.json")):
            with open(pth, 'rb') as f_json:
                lines = f_json.readlines()
                labels.extend(lines)
        return labels
    
    def load_out_ser(self):
        data_path = self.lightgbm_config["out_ser_path"]
        labels = []
        with open(data_path, 'rb') as f_res_ser:
            lines = f_res_ser.readlines()
            labels.extend(lines)
        return labels
    
    def load_scaler(self):
        print('Loading scaler post processing relation ...')
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f_scaler:
                scaler = pickle.load(f_scaler)
            f_scaler.close()
            return scaler 
        else:
            print("Path to scaler not exist !")
            
    def load_ser2re_data(self):
        data_path = self.lightgbm_config["out_ser_path"]
        labels = []
        with open(data_path, 'rb') as f_res_ser:
            lines = f_res_ser.readlines()
            labels.extend(lines)
        return labels
    
    def preprocess_ser2re_batch_ver2(self, im_path, data = None):
        h, w = cv2.imread(im_path).shape[:2]
        f_name = os.path.basename(im_path)
        df_d = pd.DataFrame(data)
        df_d['width'] = w
        df_d['height'] = h
        df_d['fname'] = f_name
        re = []
        df_key = df_d[df_d.pred.str.lower()=='key']
        if df_key.shape[0] == 0:
            df_key = df_d[df_d.pred.str.lower()=='question']
        if df_key.shape[0] == 0:
            print("No key was found")
            re = pd.DataFrame(re)
            return re
        df_key_transcription = df_key.transcription.values.tolist()
        df_key_transcription = self.kv_embed.embedding(df_key_transcription)
        df_key["embedding"] = df_key_transcription.tolist()
        
        df_value = df_d[df_d.pred.str.lower()=='value']
        if df_value.shape[0] == 0:
            df_value = df_d[df_d.pred.str.lower()=='answer']
        if df_value.shape[0] == 0:
            print("No value was found")
            re = pd.DataFrame(re)
            return re
        df_value_transcription = df_value.transcription.values.tolist()
        df_value_transcription = self.kv_embed.embedding(df_value_transcription)
        df_value["embedding"] = df_value_transcription.tolist()
        
        for key in df_key.iterrows():
            linking = key[-1].linking
            for value in df_value.iterrows():
                if [key[-1].id, value[-1].id] in linking:
                    link_label =1.0
                else:
                    link_label =0.0
                re.append({
                    'k_id': key[-1].id,
                    'k_text': str(key[-1].transcription),
                    'k_embed': key[-1].embedding,
                    'k_box': points2xyxy(key[-1].points),
                    'v_id': value[-1].id,
                    'v_text': value[-1].transcription,
                    'v_embed': value[-1].embedding,
                    'v_box': points2xyxy(value[-1].points),
                    'width': w,
                    'height': h,
                    'fname': os.path.basename(f_name),
                    'label': link_label
                })
        re = pd.DataFrame(re)
        if re.shape[0] == 0:
            print("No question-answer pair was found")
            return re
    
        return re.reset_index(drop=True)
    
    def post_process(self, df: pd.DataFrame, pred_prob, threshold = 0):
        # one value only links to one key but one key can link to many value
        df['pred_prob'] = pred_prob
        df['is_linking'] = 0.0
        fnames = df.fname.unique().tolist()
        for fname in fnames:
            df_fname = df[df.fname==fname]
            v_ids = df_fname.v_id.unique().tolist()
            for v_id in v_ids:
                df_vid = df_fname[df_fname.v_id==v_id]
                idx_max = df_vid.pred_prob.idxmax()
                if df.loc[(df.fname==fname)&(df.v_id==v_id)&(df.index==idx_max), 'pred_prob'].values[0] >= threshold:
                    df.loc[(df.fname==fname)&(df.v_id==v_id)&(df.index==idx_max), 'is_linking'] = 1.0

        return df
    
            
    def make_features(self, df:pd.DataFrame):
        """Create feature from dataframe

        Args:
            df (pd.DataFrame): input data

        Returns:
            pd.DataFrame: feature after process
        """
        df = df[self.cols + ['label']]
        df.k_box = df.apply(lambda x: normalize_scale_bbox(x.k_box, x.width, x.height), axis=1)
        df.v_box = df.apply(lambda x:normalize_scale_bbox(x.v_box, x.width, x.height), axis=1)
        k_features = pd.DataFrame(df.k_box.tolist(), index=df.index, columns=['k_' + s for s in ['x1', 'y1', 'x2', 'y2']])
        v_features = pd.DataFrame(df.v_box.tolist(), index=df.index, columns=['v_' + s for s in ['x1', 'y1', 'x2', 'y2']])
        
        df = pd.concat([k_features, v_features, df[self.cols], df['label']], axis=1)
        
        df['k_cx'] = df.k_x1.add(df.k_x2).div(2)
        df['k_cy'] = df.k_y1.add(df.k_y2).div(2)
        
        df['v_cx'] = df.v_x1.add(df.v_x2).div(2)
        df['v_cy'] = df.v_y1.add(df.v_y2).div(2)
        
        df['fe1'] = abs(df.v_x1 - df.k_x1)
        df['fe2'] = abs(df.v_y1 - df.k_y1)
        df['fe3'] = abs(df.v_x1 - df.k_x2)
        df['fe4'] = abs(df.v_y1 - df.k_y2)
        df['fe5'] = abs(df.v_x2 - df.k_x1)
        df['fe6'] = abs(df.v_y2 - df.k_y1)
        df['fe7'] = abs(df.v_x2 - df.k_x2)
        df['fe8'] = abs(df.v_y2 - df.k_y2)
        df['fe9'] = abs(df.v_x2 - df.v_x1)
        df['fe10'] = abs(df.v_y2 - df.v_y1)
        df['fe11'] = abs(df.k_x2 - df.k_x1)
        df['fe12'] = abs(df.k_y2 - df.k_y1)
        
        df['fe13'] = df.apply(lambda x: cal_degrees([x.k_x1, x.k_y1], [x.v_x1, x.v_y1]), axis=1)
        df['fe14'] = df.apply(lambda x: cal_degrees([x.k_x2, x.k_y1], [x.v_x2, x.v_y1]), axis=1)
        df['fe15'] = df.apply(lambda x: cal_degrees([x.k_x2, x.k_y2], [x.v_x2, x.v_y2]), axis=1)
        df['fe16'] = df.apply(lambda x: cal_degrees([x.k_x1, x.k_y2], [x.v_x1, x.v_y2]), axis=1)
        df['fe17'] = df.apply(lambda x: cal_degrees([x['k_cx'], x['k_cy']], [x['v_cx'], x['v_cy']]), axis=1)
        
        df['fe18'] = df.apply(lambda x: boxes_distance([x.k_x1-x.v_x2, x.k_y2-x.v_y1],[x.v_x1-x.k_x2, x.v_y2-x.k_y1]), axis=1)
        df['fe19'] = df.apply(lambda x: dist_points([x.k_cx, x.k_cy], [x.v_cx, x.v_cy]), axis=1)
        
        # df['fe20'] = df['k_embed']
        # df['fe21'] = df['v_embed']
        
        k_embed_df = pd.DataFrame(np.array(df['k_embed'].values.tolist())).add_prefix('fe20')
        df = pd.concat([df, k_embed_df], axis=1)
        
        v_embed_df = pd.DataFrame(np.array(df['k_embed'].values.tolist())).add_prefix('fe21')
        df = pd.concat([df, v_embed_df], axis=1)
        
        cols = [c for c in df.columns if c.startswith('fe')] + ['label']

        return df[cols], df[self.cols]
    
    def predict(self, data: pd.DataFrame):
        d_features, __ = self.make_features(data)
        X, y = d_features.values[:, :-1], d_features.values[:, -1]
        X_transform = self.scaler.transform(X)
        pred_prob = self.model_classifier.predict_proba(X_transform)[:, 1]
        pred_df = self.post_process(data, pred_prob)
        return pred_df
    
    # def visualize(self, data: pd.DataFrame):
    #     print("Visualizing results ...")
    #     fname_list = data.fname.unique().tolist()
    #     # print("-------------------------------", fname_list)
    #     os.makedirs(os.path.join(self.lightgbm_config["debug_dir"], 'visualize'), exist_ok=True)
    #     for fname in tqdm(fname_list):
    #         img_pth = os.path.join("/home/chuongphung/projects/chatgpt/doi/inference_code/test_dataset/image", fname)
    #         img = cv2.imread(img_pth)
    #         img_copy = img.copy()
    #         d_fname = data[data.fname==fname]
    #         if d_fname.shape[0] > 0:
    #             img_copy = Visualization.visualize_ser_re(d_fname, img_copy)
            
    #         cv2.imwrite(os.path.join(self.lightgbm_config["debug_dir"], 'visualize', fname), img_copy)
    
    def run(self, im_path, document_ser_results):
        """Infer model lightgbm

        Args:
            im_path (str): path to image
            document_ser_results (dictionary): result of ser model

        Returns:
            dictionary: final result
        """
        # start_time = time.time()
        data = self.preprocess_ser2re_batch_ver2(im_path, document_ser_results)
        # print("----------============", time.time()-start_time)
        if len(data)==0:
            return []
        result = self.predict(data)
        # self.visualize(result)
        all_linking = []
        result_dict = {"transcription":None, "label":None, "points":None, "id": None, "linking":None, "bbox":None, "pred_id":None, "pred": None}
        linking_result = result.loc[result["is_linking"] == 1.0]
        # print("================-----document_ser_results------==================\n", document_ser_results)
        # print("----------============", linking_result.columns)
        ser_df = pd.DataFrame(document_ser_results)
        l_id = np.array(ser_df.id.values)
        l_pred_id = np.array(ser_df.pred_id.values)
        l_pred = np.array(ser_df.pred.values)
        for index, key_value in linking_result.iterrows():

            key_dict = result_dict.copy()
            value_dict = result_dict.copy()
            
            key_dict["transcription"] = key_value.k_text
            value_dict["transcription"] = key_value.v_text
            
            key_dict["points"] = key_value.k_box.tolist()
            value_dict["points"] = key_value.v_box.tolist()
            
            key_dict["id"] = key_value.k_id
            value_dict["id"] = key_value.v_id
            
            key_dict["bbox"] = key_value.k_box.tolist()
            value_dict["bbox"] = key_value.v_box.tolist()
            
            key_dict["pred_id"] = l_pred_id[np.where(l_id==key_value.k_id)[0]].tolist()[0]
            value_dict["pred_id"] = l_pred_id[np.where(l_id==key_value.v_id)[0]].tolist()[0]
            
            key_dict["pred"] = l_pred[np.where(l_id==key_value.k_id)[0]].tolist()[0]
            value_dict["pred"] = l_pred[np.where(l_id==key_value.v_id)[0]].tolist()[0]
            
            key_dict["linking"] = [[key_value.k_id, key_value.v_id]]
            value_dict["linking"] = [[key_value.k_id, key_value.v_id]]
                
            all_linking.append([key_dict, value_dict])
        # print("----------============\n", all_linking)
        return all_linking

if __name__ == "__main__":
    predictor = Predictor()
    data = predictor.preprocess_ser2re_batch()
    result = predictor.predict(data)
    print("------------------------------------\n", result[["label", "is_linking"]])
    print("================f1_score: ", f1_score(result.label.values, result.is_linking.values))
    print("================precision_score: ", precision_score(result.label.values, result.is_linking.values))
    print("================recall_score: ", recall_score(result.label.values, result.is_linking.values))