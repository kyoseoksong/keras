# -*- coding: utf8 -*-

'''
Keras를 이용하여 원하는 오브젝트를 classify한다.

원하는 Pre-trained 모델을 바로 불러와서 사용할 수 있다
    ResNet50, InceptionV3, Xception, VGG16, VGG19 중 선택 가능

원하는 이미지를 바로 지정해 줄 수 있다

커맨드라인에서 사용 방법
python 2017-08-07 Pretrained models.py --image "prius.png" --model "ResNet50"
'''

import numpy as np
import argparse
# import cv2

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19

from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

# 커맨드라인에서의 인자 파싱 설정

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="입력 이미지 경로를 지정해 주세요")
ap.add_argument("-model", "--model", type=str, default="VGG16", help="사용할 pre-trained network 을 지정해 주세요")
args = vars(ap.parse_args())

# 모델 사전 정의

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50,
}

if args["model"] not in MODELS.keys():
    raise AssertionError("방금 지정한 모델은 'MODELS' 사전에 포함된 모델이 아닙니다")

# 입력 이미지의 크기 초기화

# VGG나 ResNet의 경우에는 사이즈가 (224,224)이며 표준 전처리를 해 준다.
inputShape = (224,224)
preprocess = imagenet_utils.preprocess_input

# Inception이나 Xception인 경우에는 사이즈가 (299,299)이며 별도의 전처리를 해 준다.
if args["model"] in ("inception", "xception"):
    inputShape = (299,299)
    preprocess = preprocess_input

# 디스크에 저장되어 있는 Pre-trained model의 weight를 로드하여 우리의 모델을 인스턴스화해 준다.

print "[INFO] 미리 학습된 '{}' 모델을 로드하고 있습니다...".format(args["model"])
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# Classifiy할 이미지를 준비해 주자

print "[INFO] '{}' 이미지를 로드하여 전처리하고 있습니다...".format(args["image"])
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image) # (224,224,3)
image = np.expand_dims(image, axis=0) # (1,224,224,3)
image = preprocess(image) # 전처리 (평균 빼주기, 스케일링)

# 이제 이미지를 classify해 주자

print "[INFO] 이미지를 분류하고 있습니다...".format(args["model"])
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

for i, (imagenetID, label, prob) in enumerate(P[0]):
    print "{}. {}: {:.2f}%".format(i+1, label, prob*100)

# 이미지 보여주기 (OpenCV2 설치되어 있어야 한다)
'''    
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob*100), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
'''



















