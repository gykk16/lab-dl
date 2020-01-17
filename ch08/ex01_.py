"""
현재 버전보다 설치된 버전이 낮은 패키지를 찾을 때
    pip list --outdated
설치된 패키지를 업데이트 할 때
    pip install --upgrade 패키지1, 패키지2, ...
TensorFlow 설치
    pip install tensorflow  (GPU 사용하지 않는 버전)
    pip install tensorflow-gpu  (GPU 사용하는 버전)
Keras 설치
    pip install keras
"""

import tensorflow as tf
import keras

print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)
