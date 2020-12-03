import numpy as np
import matplotlib
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import keras
import itertools
from keras.models import model_from_json
import os
from skimage.feature import local_binary_pattern, draw_multiblock_lbp, multiblock_lbp
from tqdm import tqdm
import cv2
import pickle
from sklearn.preprocessing import MinMaxScaler
from lbp_classification.yoloface.utils import apply_yolo_face

test_filenames = ['Liu100.jpg', 'Liu72.jpg', 'Liu104.jpg', 'Liu103.jpg', 'Liu88.jpg', 'Liu20.jpg', 'Liu123.jpg', 'Liu153.jpg', 'Liu163.jpg', 'Liu50.jpg', 'Liu1.jpg', 'Liu47.jpg', 'Liu37.jpg', 'Liu105.jpg', 'Liu6.jpg', 'Liu121.jpg', 'Liu149.png', 'Liu35.jpg', 'Liu56.jpg', 'Liu51.jpg', 'Liu61.jpg', 'Liu127.jpg', 'Liu160.jpg', 'Liu114.jpg', 'Liu8.jpg', 'Liu131.jpg']
filename_pbar = tqdm(test_filenames, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
image_source_folder = 'F:/Invisible Man/Images/Studio_Filtered/'
for f in filename_pbar:
    filename_pbar.set_description("Processing %s" % f)
    color = cv2.imread(image_source_folder + f)
    img = cv2.imread(image_source_folder + f, 0)
    faces, confidences, plotted_frame = apply_yolo_face(color, 0)
    filename_pbar.set_description("Processing {} | {} faces".format(f, len(faces)))
    if len(faces) > 1 or len(faces) == 0:
        continue
    cv2.imwrite('facial_rec/'+f, plotted_frame)


