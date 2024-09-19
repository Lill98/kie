import sys
from os import makedirs
from os.path import abspath, dirname, join, basename
from typing import Dict, List, Tuple

import yaml

# NOTE: リポジトリのルートを実行ディレクトリとした場合でもfrom key_value_extractorできるようにする.
if __name__ == "__main__":
    sys.path.append("../")
from key_value_extractor.extract_ser_re import Inference, preprocess
from key_value_extractor.tools.infer_kie_token_ser import SerPredictor
from key_value_extractor.LightGBM.predict import Predictor

MODULE_ABS_PATH = dirname(abspath(__file__))

class KeyValueExtractor:
    def __init__(self):
        """
        推論に必要なインスタンスを初期化する.
        """
        # NOTE: moduleで呼び出すときにargparseの引数を指定する方法.
        # The argparse module actually reads input variables from special variable, which is called ARGV (short from ARGument Vector).
        # This variable is usually accessed by reading sys.argv from sys module.
        # https://stackoverflow.com/questions/39853278/how-to-set-argparse-arguments-from-python-script

        # WARNING: python strucuted.pyx -ap 1などでオプション引数を指定して構造化を実行すると, 
        # その引数がここで含まれてしまうので, これを回避するために代入している.
        # この後でsys.argvを利用する処理がある場合, 影響する可能性がある.
        sys.argv = [__name__]

        ser_config, xgboost_config, device = preprocess()

        makedirs(ser_config['Global']['save_res_path'], exist_ok=True)
        # os.makedirs(ser_config['Global']['save_res_path']+ "/ser", exist_ok=True)
        makedirs(ser_config['Global']['save_res_path']+ "/re", exist_ok=True)
        ser_config["Architecture"]["Backbone"]["checkpoints"] = join(MODULE_ABS_PATH, ser_config["Architecture"]["Backbone"]["checkpoints"])
        ser_config["PostProcess"]["class_path"] = join(MODULE_ABS_PATH, ser_config["PostProcess"]["class_path"])
        ser_config["Eval"]["dataset"]["transforms"][1]["VQATokenLabelEncode"]["class_path"] = join(MODULE_ABS_PATH, ser_config["Eval"]["dataset"]["transforms"][1]["VQATokenLabelEncode"]["class_path"])
        xgboost_config["scaler_path"] = join(MODULE_ABS_PATH, xgboost_config["scaler_path"])
        xgboost_config["model_path"] = join(MODULE_ABS_PATH, xgboost_config["model_path"])
        
        ser_engine = SerPredictor(ser_config)

        XG_Boost_engine = Predictor(xgboost_config)
        
        self.infer = Inference(ser_engine, XG_Boost_engine, ser_config)

    def run_inference(
        self,
        key_value_input_dict: List[Dict],
        img_file_name: str,
        img_dir: str,
    ) -> Tuple[List, List[Dict]]:
        """
        ser(semantic entity recognition), re(relation extraction)モデルの推論を実行する.

        Args:
            - key_value_input_dict( `List[Dict]` ):
            - img_file_name( `str` ):
            - img_dir( `str` ):

        Returns:

        """
        # convert input data format
        if key_value_input_dict["image_name"] != img_file_name:
            key_value_input_dict["image_name"] = img_file_name
        img_file_path = img_dir + '/' + img_file_name
        # inference
        ser_results, re_results = self.infer.run(key_value_input_dict, img_file_path)
        return re_results
