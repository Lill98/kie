## 1. Information Extraction

An approach to extract key information and relation from document

## 2. Requirement:

- python 3.8.11
- torch 1.7.1
- cuda 10.1
- paddlepaddle-gpu 2.3.2.post101
- paddlenlp==2.5.2
- transformers==4.30.1
- sentencepiece==0.1.99
- opencv-contrib-python <=4.6.0.66
- opencv <=4.6.0.66
- fugashi==1.3.0
- ipadic==1.0.0
- lightgbm
- scikit-learn==1.3.2
- shapely
- scikit-image
- imgaug
- pyclipper
- lmdb
- tqdm
- numpy
- visualdl
- rapidfuzz
- cython
- lxml
- premailer
- openpyxl
- attrdict
- PyMuPDF<1.21.0
- Pillow<=9.5.0
- yacs
- seqeval
- pypandoc
- attrdict3
- python_docx

## 3. Install packages
```
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
```
### 3.1. Install pyenv (ignore if it have already)

```
curl https://pyenv.run | bash
```


Paste
```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
```
to `~/.bashrc`.

Restart shell:
```
exec "$SHELL"
```

### 3.2. Install python 3.8.11
```
pyenv install 3.8.11

pyenv virtualenv 3.8.11 information_extraction
```

Activate environment
```
pyenv shell information_extraction

pyenv activate information_extraction
```

### 3.3. Install libs:
```
pip install -r requirements.txt

```

## 4. Testing
### 4.1 Prepare input
You need to prepare input dataset with image and json file as in test_dataset folder  
- Images: Put all image in the images folder
- Iput OCR: Put all json file in the annotations folder with corresponding name in Images folder.   The content of json file follow this structure
```
{
  "image_name": [NAME OF IMAGE IN THE IMAGES FOLDER]
  "ocr":[
    {
      "box1": [
                1534,
                15,
                2301,
                147
            ],
      "text1": "御請求書"
    },
    {
    "box2": [
                1534,
                15,
                2301,
                147
            ],
      "text2": "御請求書"
    }
  ...
  ]
}
```
Where:
- box: Your output text bounding box
- text: Content of box

### 4.2 Output 
The image and output json will be save in the save_res_path config 

The content of output json as bellow:  
```
[
    {
        "transcription": [ocr txt],
        "id": id of key1,
        "linking": [
            [
                id of value1,
                id of value2,
                .
                .
                .
            ]
        ],
        "bbox": [text bounding box],
        "pred": [label prediction]
    },
  ...
]
```
with 
+ linking: list of linking id
+ pred: predict label
### 4.3 How to run
cd to the one higher level folder. 
Run:
```
python3 [name_of_folder]/extract_ser_re.py \
  -o Global.save_res_path=[PATH to visualized output folder] \
  Global.image_folder=[PATH to input image folder] \
  Global.ocr_file=[PATH to Infordio's OCR results] \
```


Where:
- Global.save_res_path: PATH to output 
- Global.image_folder: PATH to input image 
- Global.ocr_file: PATH to Infordio's OCR results

you also can change  image_folder, ocr_file,save_res_path in file configs/ser_vi_layoutxlm_xfund_key_value.yml at row 20, 21, 22 respectively. Remember to use relative path from one higher level folder.
Example:
run 
```
  python3 Relation%20Extraction/extract_ser_re.py \
 -o Global.image_folder=Relation%20Extraction/test_dataset/images/table_document_003.jpg \
  Global.ocr_file=Relation%20Extraction/test_dataset/annotations/table_document_003.json \
  Global.save_res_path=Relation%20Extraction/output/test 
  
```



or run below if you had config the path at file configs/ser_vi_layoutxlm_xfund_key_value.yml
```
  python3 Relation%20Extraction/extract_ser_re.py   
```

## 5. Reference:

- Paddle Key Information Extraction - Relation Extraction
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/kie/README.md