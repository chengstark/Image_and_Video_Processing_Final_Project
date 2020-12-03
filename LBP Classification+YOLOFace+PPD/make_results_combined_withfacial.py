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
import json

test_filenames = ['Liu100.jpg', 'Liu72.jpg', 'Liu104.jpg', 'Liu103.jpg', 'Liu88.jpg', 'Liu20.jpg', 'Liu123.jpg', 'Liu153.jpg', 'Liu163.jpg', 'Liu50.jpg', 'Liu1.jpg', 'Liu47.jpg', 'Liu37.jpg', 'Liu105.jpg', 'Liu6.jpg', 'Liu121.jpg', 'Liu149.png', 'Liu35.jpg', 'Liu56.jpg', 'Liu51.jpg', 'Liu61.jpg', 'Liu127.jpg', 'Liu160.jpg', 'Liu114.jpg', 'Liu8.jpg', 'Liu131.jpg']
step = 20
size = 120
radius = 2
n_points = 8 * radius
image_source_folder = 'F:/Invisible Man/Images/Studio_Filtered/'
with open('mask_bounding_box.json', 'r') as f:
    ground_truth_bbox_dict = json.load(f)

def get_mask_density(img):
    train_filenames =  ['Liu137.jpg', 'Liu97.jpg', 'Liu78.jpg', 'Liu14.jpg', 'Liu69.jpg', 'Liu164.jpg', 'Liu138.jpg', 'Liu167.jpg', 'Liu107.jpg', 'Liu19.jpg', 'Liu52.jpg', 'Liu48.jpg', 'Liu5.jpg', 'Liu87.jpg', 'Liu148.jpg', 'Liu60.jpg', 'Liu116.jpg', 'Liu65.jpg', 'Liu168.jpg', 'Liu71.jpg', 'Liu22.jpg', 'Liu145.jpg', 'Liu112.jpg', 'Liu108.jpg', 'Liu141.jpg', 'Liu55.jpg', 'Liu117.jpg', 'Liu125.jpg', 'Liu73.jpg', 'Liu57.jpg', 'Liu23.jpg', 'Liu99.jpg', 'Liu152.jpg', 'Liu31.png', 'Liu81.jpg', 'Liu144.png', 'Liu92.jpg', 'Liu140.jpg', 'Liu82.jpg', 'Liu58.jpg', 'Liu95.jpg', 'Liu34.jpg', 'Liu124.jpg', 'Liu80.jpg', 'Liu77.jpg', 'Liu130.jpg', 'Liu106.jpg', 'Liu170.jpg', 'Liu171.jpg', 'Liu40.jpg', 'Liu85.jpg', 'Liu66.jpg', 'Liu4.jpg', 'Liu132.jpg', 'Liu84.jpg', 'Liu115.jpg', 'Liu83.jpg', 'Liu32.png', 'Liu39.jpg', 'Liu43.jpg', 'Liu62.jpg', 'Liu28.jpg', 'Liu13.jpg', 'Liu46.jpg', 'Liu162.jpg', 'Liu122.jpg', 'Liu10.jpg', 'Liu126.jpg', 'Liu36.jpg', 'Liu26.jpg', 'Liu119.jpg', 'Liu18.jpg', 'Liu136.jpg', 'Liu79.jpg', 'Liu155.jpg', 'Liu101.jpg', 'Liu157.jpg', 'Liu29.jpg', 'Liu75.jpg', 'Liu161.jpg', 'Liu118.jpg', 'Liu111.jpg', 'Liu96.jpg', 'Liu64.jpg', 'Liu109.jpg', 'Liu143.jpg', 'Liu25.jpg', 'Liu165.jpg', 'Liu142.jpg', 'Liu12.jpg', 'Liu41.jpg', 'Liu89.jpg', 'Liu147.jpg', 'Liu128.jpg', 'Liu156.jpg', 'Liu102.jpg', 'Liu38.jpg', 'Liu146.jpg', 'Liu42.jpg', 'Liu158.jpg', 'Liu17.jpg', 'Liu133.jpg', 'Liu135.jpg', 'Liu9.jpg', 'Liu24.jpg', 'Liu59.jpg', 'Liu15.jpg', 'Liu16.jpg', 'Liu3.jpg', 'Liu93.jpg', 'Liu139.jpg', 'Liu33.jpg', 'Liu113.jpg', 'Liu134.jpg', 'Liu86.jpg', 'Liu30.jpg', 'Liu2.jpg', 'Liu120.jpg', 'Liu11.jpg', 'Liu21.jpg', 'Liu68.jpg', 'Liu166.jpg', 'Liu63.jpg', 'Liu44.jpg', 'Liu90.jpg', 'Liu76.jpg', 'Liu54.jpg', 'Liu169.jpg', 'Liu53.jpg', 'Liu70.jpg', 'Liu67.jpg', 'Liu27.jpg', 'Liu154.jpg', 'Liu94.png', 'Liu49.jpg', 'Liu151.jpg', 'Liu7.jpg', 'Liu98.jpg', 'Liu45.jpg', 'Liu150.jpg', 'Liu91.jpg']
    dim = (img.shape[1], img.shape[0])
    mask_density = np.zeros_like(img).astype(np.float32)
    for f in train_filenames:
        mask = cv2.imread('F:/Invisible Man/Images/Studio_Masks/'+f, 0)
        mask.astype(np.float32)
        mask = mask / 255
        resized_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
        mask_density += resized_mask

    kernel = np.ones((3,3),np.float32)
    mask_density = cv2.filter2D(mask_density,-1,kernel)
    mask_density /= np.max(mask_density)
    return mask_density


keras_model_name = 'model_auc'
json_file = open('{}.json'.format(keras_model_name), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("{}.h5".format(keras_model_name))
print("Loaded model from disk")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.FalsePositives()])

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

face_names = ['Liu104.jpg', 'Liu88.jpg', 'Liu20.jpg', 'Liu50.jpg', 'Liu1.jpg', 'Liu149.png', 'Liu35.jpg', 'Liu8.jpg']
threshs = []
filename_pbar = tqdm(test_filenames, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
throw_away_count = 0
face_boosted_images = []
weight_facial_rec = True
ious = []
for f in filename_pbar:
    face_detected = False
    X_test = []
    vote_recorder = []
    filename_pbar.set_description("Processing %s" % f)
    color = cv2.imread(image_source_folder + f)
    img = cv2.imread(image_source_folder + f, 0)
    mask_density = get_mask_density(img)
    tmp = mask_density.copy()
    faces, face_confidences, plotted_frame = apply_yolo_face(color, 0.4)
    if len(faces) == 1:
        cv2.imwrite('test_results/' + f.split('.')[0] + '_facial_rec.png', plotted_frame)
        face_x, face_y, face_w, face_h = faces[0]
        face_detected = True
        if weight_facial_rec:
            mask_density[face_y-face_h: face_y+face_h*10, face_x-(face_w):face_x+(face_w*2)] *= (1+face_confidences[0])
            # print(f, 1+face_confidences[0], np.sum(mask_density) - np.sum(tmp))
            # print(mask_density.shape, face_y, face_h*10, face_x-face_w, face_x+(face_w*2))
            mask_density /= np.max(mask_density)
            face_boosted_images.append(f)
            filename_pbar.set_description("Processing %s" % f + ' | facial boost {} calculated'.format(1+face_confidences[0]))
        else:
            result6 = color.copy()
            result6 = cv2.rectangle(result6, (face_x-(face_w), face_y-face_h),
                                    (face_x+(face_w*2),face_y+(face_h*10)), (0, 0, 255), thickness=3)
            face_boosted_images.append(f)
            cv2.imwrite('test_results/' + f.split('.')[0] + '_result_final_face.png', result6)
            ground_truth_bbox = ground_truth_bbox_dict[f]
            iou = calc_iou([face_x-(face_w), face_y-face_h, face_x+(face_w*2),face_y+(face_h*10)],
                           [ground_truth_bbox[0], ground_truth_bbox[1], ground_truth_bbox[0]+ground_truth_bbox[2], ground_truth_bbox[1]+ground_truth_bbox[3]])
            ious.append(iou)
            continue
    else:
        filename_pbar.set_description("Processing %s" % f + ' | mask density calculated ')
        # continue

    plt.imshow(mask_density, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig('test_results/' + f.split('.')[0] + '_ppd.png', bbox_inches='tight')
    plt.clf()

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
    percentile_thresh, cluster_boundary_thresh = get_threshold(preds, percentile=50)
    thresh = percentile_thresh
    threshs.append(thresh)
    filename_pbar.set_description("Processing %s" % f + ' | threshold {}'.format(thresh))

    pure_prediction_overlay = np.zeros(img.shape).astype(np.uint8)

    vote_recorder = np.asanyarray(vote_recorder)
    #     confidence_recorder = np.zeros_like(vote_recorder).astype(np.float32)
    confidence_recorder = np.zeros((len(range(0, img.shape[0], step)), len(range(0, img.shape[1], step))))
    # classify each window
    idx = 0
    y_idx = 0
    for y in range(0, img.shape[0], step):
        x_idx = 0
        for x in range(0, img.shape[1], step):
            window = img[y:y + size, x:x + size]
            if window.shape[0] == size and window.shape[1] == size:
                pred = preds[idx]
                if pred > thresh:
                    vote_recorder[y_idx][x_idx] += 1
                    density_weighted_confidence = pred * np.sqrt(mask_density[y][x])
                    confidence_recorder[y_idx][x_idx] += density_weighted_confidence
                    overlay = cv2.rectangle(overlay, (x, y), (x + size, y + size), (0, 0, 255), thickness=-1)
                    overlay = cv2.rectangle(overlay, (x, y), (x + size, y + size), (0, 0, 0), thickness=3)
                    pure_prediction_overlay = cv2.rectangle(pure_prediction_overlay, (x, y), (x + size, y + size), 255,
                                                            thickness=-1)

                idx += 1
                x_idx += 1

        y_idx += 1
    result = cv2.addWeighted(base, 1.0, overlay, 0.5, 1)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_pred.png', result)

    #     filename_pbar.set_description("Processing %s" % f+' | calculating Bayes ')

    #     bayes_results = []
    #     bayes_overlay_bw = np.zeros(img.shape).astype(np.float32)
    #     for y in range(vote_recorder.shape[0]):
    #         bayes_row = []
    #         for x in range(vote_recorder.shape[1]):
    #             if x-1>= 0 and x+1< vote_recorder.shape[1] and y-1>=0 and y+1 < vote_recorder.shape[0]:
    # #                 feature = np.asarray([[x*step, y*step, vote_recorder[y-1][x], vote_recorder[y+1][x], vote_recorder[y][x-1], vote_recorder[y][x+1]]]).astype(np.float64)
    #                 feature = np.asarray([[x*step, y*step]]).astype(np.float64)
    #                 bayes_pred = bayes.predict(feature)[0]
    #                 bayes_row.append(bayes_pred)

    #         if len(bayes_row) > 0:
    #             bayes_results.append(bayes_row)

    #     bayes_results = np.asarray(bayes_results)
    #     filename_pbar.set_description("Processing %s" % f+' | normalizing Bayes ')
    #     bayes_results = MinMaxScaler().fit_transform(bayes_results)

    #     filename_pbar.set_description("Processing %s" % f+' | painting Bayes ')

    #     bayes_overlay_bw = np.zeros(img.shape).astype(np.float32)
    #     for y in range(vote_recorder.shape[0]):
    #         for x in range(vote_recorder.shape[1]):
    #             if x-1>= 0 and x+1< vote_recorder.shape[1] and y-1>=0 and y+1 < vote_recorder.shape[0]:
    #                 coord_x = x*step
    #                 coord_y = y*step
    #                 c = int(bayes_results[y-1][x-1]*255)
    #                 bayes_overlay_bw = cv2.rectangle(bayes_overlay_bw, (coord_x, coord_y), (coord_x + size, coord_y + size), c, thickness=-1)
    #     cv2.imwrite('test_results/' + f.split('.')[0]+'_bayes_heat.png', bayes_overlay_bw)

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

    confidence_recorder = confidence_recorder / np.max(confidence_recorder)
    unique_confidence = np.unique(confidence_recorder, return_counts=True)[0]
    if weight_facial_rec and face_detected:
        confidence_thresh = np.percentile(unique_confidence, 50)
        confidence_recorder[confidence_recorder < confidence_thresh] = np.min(unique_confidence)

    cmap_norm = matplotlib.colors.Normalize(vmin=np.min(unique_confidence), vmax=np.max(unique_confidence))
    confidence_base = color.copy().astype(np.float32)
    confidence_overlay = np.zeros_like(color)
    confidence_overlay_adder = np.zeros_like(img).astype(np.float32)
    confidence_overlay_bw = np.zeros_like(img)
    for y in range(confidence_recorder.shape[0]):
        for x in range(confidence_recorder.shape[1]):
            neighbour_x = x - n_neighbours
            confidence = confidence_recorder[y][x]
            coord_y = y * step
            coord_x = x * step

            cmap = matplotlib.cm.get_cmap('rainbow')
            rgba = cmap(confidence)
            c = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            c_bw = int(confidence > 0) * 255
            confidence_overlay_adder[coord_y:coord_y + size, coord_x:coord_x + size] += confidence

            confidence_overlay = cv2.rectangle(confidence_overlay, (coord_x, coord_y), (coord_x + size, coord_y + size),
                                               c, thickness=-1)
            confidence_overlay_bw = cv2.rectangle(confidence_overlay_bw, (coord_x, coord_y),
                                                  (coord_x + size, coord_y + size), c_bw, thickness=-1)

    #     cv2.imwrite('test_results/' + f.split('.')[0]+'_ori.png', color)

    unique_votes = np.unique(neighbour_vote_recorder, return_counts=True)[0]
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
    cv2.imwrite('test_results/' + f.split('.')[0] + '_bw_heat.png', vote_overlay_bw)

    threshed_overlay = vote_overlay_bw.copy()
    threshed_overlay[threshed_overlay >= (np.percentile(unique_votes, 40) / np.max(unique_votes)) * 255] = 255
    threshed_overlay[threshed_overlay < (np.percentile(unique_votes, 40) / np.max(unique_votes)) * 255] = 0
    threshed_overlay = np.uint8(threshed_overlay)
    _, cnts, _ = cv2.findContours(threshed_overlay, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, prediction_cnts, _ = cv2.findContours(pure_prediction_overlay, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, confidence_cnts, _ = cv2.findContours(confidence_overlay_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    confidence_cnts_areas = []
    for idx, confidence_cnt in enumerate(confidence_cnts):
        area = cv2.contourArea(confidence_cnt)
        confidence_cnts_areas.append(area)
    area_thresh = np.percentile(np.asarray(confidence_cnts_areas), 85)
    max_confidence_bbox = None
    max_confidence_cnt = None
    max_normalized_confidence = -999
    all_normalized_confidences = []
    filtered_confidence_cnts = []
    for idx, confidence_cnt in enumerate(confidence_cnts):
        area = cv2.contourArea(confidence_cnt)
        if area < area_thresh: continue
        tmp_mask = np.zeros(img.shape).astype(np.float32)
        confidence_cnt_mask = cv2.drawContours(tmp_mask, [confidence_cnt], 0, 255, -1)
        # cv2.imwrite('test_results/' + f.split('.')[0] + '_{}.png'.format(idx), confidence_cnt_mask)
        confidence_sum = np.sum(confidence_overlay_adder[confidence_cnt_mask == 255])
        normalized_confidence = confidence_sum / area
        # print(idx, confidence_sum, area, normalized_confidence, max_normalized_confidence)
        #         print(np.sum(confidence_overlay_adder), confidence_sum, normalized_confidence, area)
        all_normalized_confidences.append(normalized_confidence)
        filtered_confidence_cnts.append(confidence_cnt)
        if normalized_confidence > max_normalized_confidence:
            max_normalized_confidence = normalized_confidence
            max_confidence_bbox = cv2.boundingRect(confidence_cnt)
            max_confidence_cnt = confidence_cnt

    #     selected_confidence_cnts = []
    #     for idx, confidence_cnt in enumerate(filtered_confidence_cnts):
    #         if all_normalized_confidences[idx] > np.percentile(np.asarray(all_normalized_confidences), 90):
    #             selected_confidence_cnts.append(confidence_cnt)
    #
    #     min_x1 = 999999
    #     min_y1 = 999999
    #     max_x2 = 0
    #     max_y2 = 0
    #     for selected_confidence_cnt in selected_confidence_cnts:
    #         x1, y1, w, h = cv2.boundingRect(selected_confidence_cnt)
    #         x2, y2 = x1+w, y1+h
    #         min_x1 = min(x1, min_x1)
    #         min_y1 = min(y1, min_y1)
    #         max_x2 = max(x2, max_x2)
    #         max_y2 = max(y2, max_y2)
    #
    # result5 = color.copy()
    # result5 = cv2.rectangle(result5, (min_x1, min_y1), (max_x2, max_y2), (0, 0, 255), thickness=3)
    # cv2.imwrite('test_results/' + f.split('.')[0] + '_result5.png', result5)


    confidence_overlay_wcnt = confidence_overlay.copy()
    confidence_overlay_wcnt = cv2.drawContours(confidence_overlay_wcnt, [max_confidence_cnt], 0, (255, 255, 255), 5)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_confidence_heat.png', confidence_overlay)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_confidence_heat_wcnt.png', confidence_overlay_wcnt)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_confidence_adder.png', confidence_overlay_adder)

    result4 = color.copy()
    result4 = cv2.rectangle(result4, (max_confidence_bbox[0], max_confidence_bbox[1]),
                            (max_confidence_bbox[0] + max_confidence_bbox[2],
                             max_confidence_bbox[1] + max_confidence_bbox[3]), (0, 0, 255), thickness=3)
    ground_truth_bbox = ground_truth_bbox_dict[f]
    iou = calc_iou([max_confidence_bbox[0], max_confidence_bbox[1], max_confidence_bbox[0] + max_confidence_bbox[2],
                             max_confidence_bbox[1] + max_confidence_bbox[3]],
                   [ground_truth_bbox[0], ground_truth_bbox[1], ground_truth_bbox[0] + ground_truth_bbox[2],
                    ground_truth_bbox[1] + ground_truth_bbox[3]])
    ious.append(iou)
    cv2.imwrite('test_results/' + f.split('.')[0] + '_result_final.png', result4)

    target_cnt = max(cnts, key=cv2.contourArea)
    bounding_rec = cv2.boundingRect(target_cnt)
    x, y, w, h = bounding_rec
    target_overlay_mask = np.zeros(img.shape).astype(np.float32)
    target_overlay_mask = cv2.drawContours(target_overlay_mask, [target_cnt], 0, 255, -1)
    result = color.copy()
    result = cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

    #     max_iou_bbox = None
    #     max_iou_cnt = None
    #     max_iou = -999
    #     for idx, prediction_cnt in enumerate(prediction_cnts):
    #         x2,y2,w2,h2 = cv2.boundingRect(prediction_cnt)
    #         iou = calc_iou([x, y, x+w, y+h], [x2, y2, x2+w2, y2+h2])
    #         if iou > max_iou:
    #             max_iou = iou
    #             max_iou_bbox = [x2,y2,w2,h2]
    #             max_iou_cnt = prediction_cnt

    #     result2 = color.copy()
    #     result2 = cv2.rectangle(result2, (max_iou_bbox[0], max_iou_bbox[1]),
    #                             (max_iou_bbox[0]+max_iou_bbox[2], max_iou_bbox[1]+max_iou_bbox[3]), (0, 0, 255), thickness=3)
    # result3 = color.copy()
    # result3 = cv2.drawContours(result3, [max_iou_cnt], 0, (0, 0, 255), 3)

    cv2.imwrite('test_results/' + f.split('.')[0] + '_result_intermediate.png', result)

    # overlay_masks = []
    # all_cnts = np.zeros(img.shape).astype(np.float32)
    # for idx, cnt in enumerate(cnts):
    #     overlay_mask = np.zeros(img.shape).astype(np.float32)
    #     overlay_mask = cv2.drawContours(overlay_mask, cnts, idx, 255, -1)
    #     all_cnts = cv2.drawContours(all_cnts, cnts, idx, 255, 2)
        # overlay_masks.append(overlay_mask)

    # cv2.imwrite('test_results/' + f.split('.')[0] + '_cnts.png', all_cnts)
    print(iou)

print(threshs)
print(face_boosted_images)
print(np.average(np.asarray(ious)))
print(np.std(np.asarray(ious))*2)
print(ious)
