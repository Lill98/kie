import cv2
import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
import copy
# sys.path.append(dname)

from glob import glob
import json
import paddle
import paddle.distributed as dist
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append("../")
from key_value_extractor.tools.program import ArgsParser, load_config, merge_config
from key_value_extractor.tools.infer_kie_token_ser import SerPredictor
# from tools.infer_kie_token_re import RePredictor
from key_value_extractor.ppocr.utils.visual import draw_re_results, draw_ser_results
from key_value_extractor.LightGBM.predict import Predictor
# print("dname", dname)

class ReArgsParser(ArgsParser):
    def __init__(self):
        super(ReArgsParser, self).__init__()
        # self.add_argument(
        #     "-xgboost", "--config_xgboost", help="xgboost configuration file to use")

    def parse_args(self, argv=None):
        args = super(ReArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=ser_configure_file_path."
        return args
    
def preprocess():
    """Preprocess config

    Returns:
        list: config after parsing
    """
    FLAGS = ReArgsParser().parse_args()
    FLAGS.config = os.path.join(dname, FLAGS.config)
    ser_config = load_config(FLAGS.config)
    ser_config = merge_config(ser_config, FLAGS.opt)
    FLAGS.config_xgboost = os.path.join(dname, FLAGS.config_xgboost)
    xgboost_config = load_config(FLAGS.config_xgboost)
    use_gpu = ser_config['Global']['use_gpu']

    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id) if use_gpu else 'cpu'
    device = paddle.set_device(device)
    if use_gpu:
        xgboost_config["device"] = "cuda"
    else:
        xgboost_config["device"] = "cpu"
    return ser_config, xgboost_config, device

def convert_input_format(ocr_result):
    """Convert to final result

    Args:
        ocr_result (dataframe): output of model XGBoost

    Returns:
        json: final result
    """
    annotation_1_file = []
    if len(ocr_result)==0:
        return ""
    for idx, box in enumerate(ocr_result):
        points = box["box"]
        if type(points) != list and type(points) != tuple: continue
        if len(points) != 4: continue
        text = box["text"]
        if type(text) != str: continue
        points = [[points[0], points[1]],
                [points[2], points[1]],
                [points[2], points[3]],
                [points[0], points[3]]]
        annotation_1_file.append({"transcription": text, "label": "other", "points": points, "id": idx, "linking": []})
    if len(annotation_1_file) == 0:
        return ""
    return json.dumps(annotation_1_file, ensure_ascii=False)

class Inference():
    def __init__(self, ser_engine, XG_Boost_engine, ser_config):
        self.ser_engine = ser_engine
        self.XG_Boost_engine = XG_Boost_engine
        self.ser_config = ser_config
        
    def merge_ser_re(self):
        id_liking = {}
        self.merge_re_result = copy.deepcopy(self.ser_results[0])
        for pairs in self.re_results:
            for pair in pairs:
                id_key = pair["id"]
                linking = pair["linking"]
                linking[0].remove(id_key)
                if id_key not in id_liking:
                    id_liking[id_key] = linking[0]
                else:
                    id_liking[id_key].extend(linking[0])
                
        for element in self.merge_re_result:
            if element["id"] in id_liking:
                element["linking"] = id_liking[element["id"]]
            for key in ["points", "pred_id", "label"]:
                element.pop(key)
            if element["pred"] == "O":
                element["pred"] = "OTHER"
        return self.merge_re_result

    def load_ocr_file(self, ocr_file_path):
        self.ocr_file_path = ocr_file_path
        try:
            with open(ocr_file_path) as f:
                ocr_res = json.load(f)
        except Exception as e:
            print(e)
            return {}
            
        if "image_name" not in ocr_res.keys() or "ocr" not in ocr_res.keys():
            print("Input was not in correct format")
            return {}
        self.ocr_res = ocr_res
        return ocr_res
    
    def load_img_path(self, data_dir, img_file_name):
        if os.path.isdir(data_dir):
            self.img_path = os.path.join(data_dir, img_file_name)
        elif os.path.isfile(data_dir):
            self.img_path = data_dir
        return self.img_path
    
    def run(self, ocr_res, img_path):
        input_data = convert_input_format(ocr_res["ocr"])
        if input_data=="":
            print("Input was not in correct format")
            return [], []
        data = {'img_path': img_path, 'label': input_data}
        self.ser_results, ser_inputs = self.ser_engine(data)
        if "pred" not in self.ser_results[0][0].keys():
            print("No Results")
            return [], []
        self.re_results = self.XG_Boost_engine.run(img_path, self.ser_results[0])
        self.merge_re_result = self.merge_ser_re()
        return self.ser_results, self.merge_re_result
    
    def visualize(self):
        path_out_ser_json = os.path.join(self.ser_config['Global']['save_res_path']+ "/ser", os.path.basename(self.ocr_file_path))
        path_out_re_json = os.path.join(self.ser_config['Global']['save_res_path']+ "/re", os.path.basename(self.ocr_file_path))
        # with open(path_out_ser_json, "w") as ff:
        #     json.dump(self.ser_results[0], ff, ensure_ascii=False)
        with open(path_out_re_json, "w") as ff:
            json.dump(self.merge_re_result, ff, ensure_ascii=False)
            
        if self.ser_config["Global"]["visualize_output"]:
            # img_vis_ser = draw_ser_results(self.img_path, self.ser_results[0])
            # save_img_path = os.path.join(
            #     self.ser_config['Global']['save_res_path'], "ser",
            #     self.ocr_res["image_name"])
            # cv2.imwrite(save_img_path, img_vis_ser)
            # img_vis_re = draw_re_results(img_vis_ser, self.re_results)
            
            img_vis_re = draw_re_results(self.img_path, self.re_results)
            
            save_img_path = os.path.join(
                self.ser_config['Global']['save_res_path'], "re",
                self.ocr_res["image_name"])
            cv2.imwrite(save_img_path, img_vis_re)
        
def main():
    ser_config, xgboost_config, device = preprocess()
    
    os.makedirs(ser_config['Global']['save_res_path'], exist_ok=True)
    # os.makedirs(ser_config['Global']['save_res_path']+ "/ser", exist_ok=True)
    os.makedirs(ser_config['Global']['save_res_path']+ "/re", exist_ok=True)
    ser_config["Architecture"]["Backbone"]["checkpoints"] = os.path.join(dname, ser_config["Architecture"]["Backbone"]["checkpoints"])
    ser_config["PostProcess"]["class_path"] = os.path.join(dname, ser_config["PostProcess"]["class_path"])
    ser_config["Eval"]["dataset"]["transforms"][1]["VQATokenLabelEncode"]["class_path"] = os.path.join(dname, ser_config["Eval"]["dataset"]["transforms"][1]["VQATokenLabelEncode"]["class_path"])
    xgboost_config["scaler_path"] = os.path.join(dname, xgboost_config["scaler_path"])
    xgboost_config["model_path"] = os.path.join(dname, xgboost_config["model_path"])
    
    ser_engine = SerPredictor(ser_config)

    XG_Boost_engine = Predictor(xgboost_config)

    data_dir = ser_config["Global"]["image_folder"]
    
    infer = Inference(ser_engine, XG_Boost_engine, ser_config)
    
    if os.path.isdir(ser_config['Global']['ocr_file']):
        for ocr_file_path in tqdm(glob(ser_config['Global']['ocr_file']+"/*.json")):
            ocr_res = infer.load_ocr_file(ocr_file_path)
            img_path = infer.load_img_path(data_dir, ocr_res["image_name"])
            ser_results, re_results = infer.run(ocr_res, img_path)
            if len(ser_results)==0:
                continue
            infer.visualize()
    elif ser_config['Global']['ocr_file'].split(".")[-1] in ["json"]:
        ocr_res = infer.load_ocr_file(ser_config['Global']['ocr_file'])
        img_path = infer.load_img_path(data_dir, ocr_res["image_name"])
        ser_results, re_results = infer.run(ocr_res, img_path)            
        infer.visualize()
        
    else:
        print("Only support file json or folder of json")
        
        
        
        # break
    
if __name__ == "__main__":
    main()