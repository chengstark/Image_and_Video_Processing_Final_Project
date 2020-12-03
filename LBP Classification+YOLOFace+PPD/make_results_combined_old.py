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



test_filenames = ['Liu100.jpg', 'Liu72.jpg', 'Liu104.jpg', 'Liu103.jpg', 'Liu88.jpg', 'Liu20.jpg', 'Liu123.jpg', 'Liu153.jpg', 'Liu163.jpg', 'Liu50.jpg', 'Liu1.jpg', 'Liu47.jpg', 'Liu37.jpg', 'Liu105.jpg', 'Liu6.jpg', 'Liu121.jpg', 'Liu149.png', 'Liu35.jpg', 'Liu56.jpg', 'Liu51.jpg', 'Liu61.jpg', 'Liu127.jpg', 'Liu160.jpg', 'Liu114.jpg', 'Liu8.jpg', 'Liu131.jpg']
step = 20
size = 120
radius = 2
n_points = 8 * radius
image_source_folder = 'F:/Invisible Man/Images/Studio_Filtered/'

json_file = open('model_auc.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_auc.h5")
print("Loaded model from disk")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

bayes = pickle.load(open('bayes.pkl', 'rb'))


def get_threshold(preds, percentile=75):
    cluster_preds = np.asarray(preds).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_preds)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    cluster_0, cluster_1 = cluster_preds[labels == 0], cluster_preds[labels == 1]

    if np.mean(cluster_0) >= np.mean(cluster_1):
        return np.percentile(cluster_0.flatten(), 75), cluster_0[np.argmin(np.abs(cluster_0 - centers[1][0]))][0]
    else:
        return np.percentile(cluster_1.flatten(), 75), cluster_1[np.argmin(np.abs(cluster_1 - centers[0][0]))][0]


def calc_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


threshs = []
filename_pbar = tqdm(test_filenames, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
throw_away_count = 0
for f in filename_pbar:
    X_test = []
    vote_recorder = []
    filename_pbar.set_description("Processing %s" % f)
    color = cv2.imread(image_source_folder + f)
    img = cv2.imread(image_source_folder + f, 0)

    n_blocks_x = img.shape[1] // size
    n_blocks_y = img.shape[0] // size

    base = color.copy().astype(np.float32)
    overlay = np.zeros(base.shape).astype(np.float32)

    for y in range(0, img.shape[0], step):
        row_vote_recorder = []
        for x in range(0, img.shape[1], step):
            window = img[y:y + size, x:x + size]

            lbp = local_binary_pattern(window, n_points, radius, 'uniform')
            lbp_counts, _ = np.histogram(lbp, bins=np.arange(radius ** 8 + 1), density=True)

            if window.shape[0] == size and window.shape[1] == size:
                X_test.append(lbp_counts)
                row_vote_recorder.append(0)
        if len(row_vote_recorder) > 0:
            vote_recorder.append(row_vote_recorder)

    X_test = np.asarray(X_test)
    filename_pbar.set_description("Processing %s" % f + ' | lbp complete')
    # make predictions
    preds = model.predict(X_test)
    preds = preds.flatten().tolist()
    # calculate threshold
    percentile_thresh, boundary_thresh = get_threshold(preds, percentile=95)
    threshs.append(percentile_thresh)
    filename_pbar.set_description("Processing %s" % f + ' | threshold {}'.format(percentile_thresh))

    pure_prediction_overlay = np.zeros(img.shape).astype(np.uint8)

    # classify each window
    idx = 0
    y_idx = 0
    for y in range(0, img.shape[0], step):
        x_idx = 0
        for x in range(0, img.shape[1], step):
            window = img[y:y + size, x:x + size]
            if window.shape[0] == size and window.shape[1] == size:
                pred = preds[idx]
                if pred > percentile_thresh:
                    vote_recorder[y_idx][x_idx] += 1
                    overlay = cv2.rectangle(overlay, (x, y), (x + size, y + size), (0, 0, 255), thickness=-1)
                    overlay = cv2.rectangle(overlay, (x, y), (x + size, y + size), (0, 0, 0), thickness=3)
                    pure_prediction_overlay = cv2.rectangle(pure_prediction_overlay, (x, y), (x + size, y + size), 255,
                                                            thickness=-1)

                idx += 1
                x_idx += 1

        y_idx += 1
    result = cv2.addWeighted(base, 1.0, overlay, 0.5, 1)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_pred.png', result)

    vote_recorder = np.asanyarray(vote_recorder)

    filename_pbar.set_description("Processing %s" % f + ' | calculating Bayes ')

    bayes_results = []
    bayes_overlay_bw = np.zeros(img.shape).astype(np.float32)
    lens = []
    for y in range(vote_recorder.shape[0]):
        bayes_row = []
        for x in range(vote_recorder.shape[1]):
            if x - 1 >= 0 and x + 1 < vote_recorder.shape[1] and y - 1 >= 0 and y + 1 < vote_recorder.shape[0]:
                feature = np.asarray([[x * step, y * step, vote_recorder[y - 1][x], vote_recorder[y + 1][x],
                                       vote_recorder[y][x - 1], vote_recorder[y][x + 1]]]).astype(np.float64)
                print(feature)
                bayes_pred = bayes.predict(feature)[0]
                print(bayes_pred)
                bayes_row.append(bayes_pred)

        if len(bayes_row) > 0:
            lens.append(len(bayes_row))
            bayes_results.append(bayes_results)
        print('row')
    print(np.unique(np.asarray(lens)))
    print('here0')
    bayes_results = np.asarray(bayes_results)
    print('here')
    filename_pbar.set_description("Processing %s" % f + ' | normalizing Bayes ')
    print('here 2')
    bayes_results = MinMaxScaler().fit_transform(bayes_results)
    print('here 3')

    filename_pbar.set_description("Processing %s" % f + ' | painting Bayes ')

    bayes_overlay_bw = np.zeros(img.shape).astype(np.float32)
    for y in range(vote_recorder.shape[0]):
        for x in range(vote_recorder.shape[1]):
            if x - 1 >= 0 and x + 1 < vote_recorder.shape[1] and y - 1 >= 0 and y + 1 < vote_recorder.shape[0]:
                coord_x = x * step
                coord_y = y * step
                c = int(bayes_results[y][x] * 255)
                bayes_overlay_bw = cv2.rectangle(bayes_overlay_bw, (coord_x, coord_y), (coord_x + size, coord_y + size),
                                                 c, thickness=-1)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_bayes_heat.png', bayes_overlay_bw)
    break
    filename_pbar.set_description("Processing %s" % f + ' | calculating neigbour votes ')

    n_neighbours = 3  # smaller values
    neighbour_vote_recorder = []
    for y in range(n_neighbours, vote_recorder.shape[0] - n_neighbours):
        row_neighbour_vote = []
        for x in range(n_neighbours, vote_recorder.shape[1] - n_neighbours):
            row_neighbour_vote.append(
                np.sum(vote_recorder[y - n_neighbours:y + n_neighbours + 1, x - n_neighbours:x + n_neighbours + 1]))
        neighbour_vote_recorder.append(row_neighbour_vote)
    neighbour_vote_recorder = np.asanyarray(neighbour_vote_recorder)

    unique_votes = np.unique(neighbour_vote_recorder, return_counts=True)[0]
    cmap_norm = matplotlib.colors.Normalize(vmin=np.min(unique_votes), vmax=np.max(unique_votes))
    vote_base = color.copy().astype(np.float32)
    vote_overlay = np.zeros(vote_base.shape).astype(np.float32)
    #     for y in range(vote_recorder.shape[0]):
    #         if y - n_neighbours < 0 or y + n_neighbours > vote_recorder.shape[0] - 1: continue
    #         for x in range(vote_recorder.shape[1]):
    #             if x - n_neighbours < 0 or x + n_neighbours > vote_recorder.shape[1] - 1: continue
    #             neighbour_y = y - n_neighbours
    #             neighbour_x = x - n_neighbours
    #             vote = neighbour_vote_recorder[neighbour_y][neighbour_x]

    #             coord_y = y*step
    #             coord_x = x*step

    #             cmap = matplotlib.cm.get_cmap('rainbow')
    #             rgba = cmap(vote)
    #             c = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
    #             vote_overlay = cv2.rectangle(vote_overlay, (coord_x, coord_y), (coord_x + size, coord_y + size), c, thickness=-1)

    #     result = cv2.addWeighted(vote_base, 1.0, vote_overlay, 0.5, 1)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_ori.png', color)
    # cv2.imwrite('test_results/' + f.split('.')[0]+'_heat.png', vote_overlay)

    vote_overlay_bw = np.zeros(img.shape).astype(np.float32)
    for y in range(vote_recorder.shape[0]):
        if y - n_neighbours < 0 or y + n_neighbours > vote_recorder.shape[0] - 1: continue
        for x in range(vote_recorder.shape[1]):
            if x - n_neighbours < 0 or x + n_neighbours > vote_recorder.shape[1] - 1: continue
            neighbour_y = y - n_neighbours
            neighbour_x = x - n_neighbours
            vote = neighbour_vote_recorder[neighbour_y][neighbour_x]

            coord_y = y * step
            coord_x = x * step

            c = int(255 * (vote / np.max(unique_votes)))
            vote_overlay_bw = cv2.rectangle(vote_overlay_bw, (coord_x, coord_y), (coord_x + size, coord_y + size), c,
                                            thickness=-1)

    threshed_overlay = vote_overlay_bw.copy()
    threshed_overlay[threshed_overlay >= (np.percentile(unique_votes, 50) / np.max(unique_votes)) * 255] = 255
    threshed_overlay[threshed_overlay < (np.percentile(unique_votes, 50) / np.max(unique_votes)) * 255] = 0
    threshed_overlay = np.uint8(threshed_overlay)
    _, cnts, _ = cv2.findContours(threshed_overlay, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, prediction_cnts, _ = cv2.findContours(pure_prediction_overlay, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    target_cnt = max(cnts, key=cv2.contourArea)
    bounding_rec = cv2.boundingRect(target_cnt)
    x, y, w, h = bounding_rec

    max_iou_bbox = None
    max_iou_cnt = None
    max_iou = -999
    for idx, prediction_cnt in enumerate(prediction_cnts):
        x2, y2, w2, h2 = cv2.boundingRect(prediction_cnt)
        iou = calc_iou([x, y, x + w, y + h], [x2, y2, x2 + w2, y2 + h2])
        if iou > max_iou:
            max_iou = iou
            max_iou_bbox = [x2, y2, w2, h2]
            max_iou_cnt = prediction_cnt

    target_overlay_mask = np.zeros(img.shape).astype(np.float32)
    target_overlay_mask = cv2.drawContours(target_overlay_mask, [target_cnt], 0, 255, -1)
    result = color.copy()
    result = cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
    result2 = color.copy()
    result2 = cv2.rectangle(result2, (max_iou_bbox[0], max_iou_bbox[1]),
                            (max_iou_bbox[0] + max_iou_bbox[2], max_iou_bbox[1] + max_iou_bbox[3]), (0, 0, 255),
                            thickness=3)
    result3 = color.copy()
    result3 = cv2.drawContours(result3, [max_iou_cnt], 0, (0, 0, 255), 3)

    cv2.imwrite('test_results/' + f.split('.')[0] + '_result.png'.format(i), result)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_result2.png'.format(i), result2)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_result3.png'.format(i), result3)
    #     cv2.imwrite('test_results/' + f.split('.')[0]+'_target.png'.format(i), target_overlay_mask)

    overlay_masks = []
    all_cnts = np.zeros(img.shape).astype(np.float32)
    for idx, cnt in enumerate(cnts):
        overlay_mask = np.zeros(img.shape).astype(np.float32)
        overlay_mask = cv2.drawContours(overlay_mask, cnts, idx, 255, -1)
        all_cnts = cv2.drawContours(all_cnts, cnts, idx, 255, 2)
        #         cv2.imwrite('test_results/' + f.split('.')[0]+'_{}.png'.format(idx), overlay_mask)
        overlay_masks.append(overlay_mask)

    cv2.imwrite('test_results/' + f.split('.')[0] + '_cnts.png', all_cnts)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_bw_heat.png', vote_overlay_bw)

print(threshs)