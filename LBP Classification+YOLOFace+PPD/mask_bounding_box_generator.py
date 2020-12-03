import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import json
# testfiles =  ['Liu100.jpg', 'Liu72.jpg', 'Liu104.jpg', 'Liu103.jpg', 'Liu88.jpg', 'Liu20.jpg', 'Liu123.jpg', 'Liu153.jpg', 'Liu163.jpg', 'Liu50.jpg', 'Liu1.jpg', 'Liu47.jpg', 'Liu37.jpg', 'Liu105.jpg', 'Liu6.jpg', 'Liu121.jpg', 'Liu149.png', 'Liu35.jpg', 'Liu56.jpg', 'Liu51.jpg', 'Liu61.jpg', 'Liu127.jpg', 'Liu160.jpg', 'Liu114.jpg', 'Liu8.jpg', 'Liu131.jpg']

bounding_boxes = dict()
with open('F:/Invisible Man/new_mask_labels.json', 'r') as f:
    mask_labels = json.load(f)

mask_labels_pbar = tqdm(mask_labels)
for row in mask_labels_pbar:
    img_name = row['External ID']
    dataset_name = row['Dataset Name']
    if dataset_name == 'Liu_Bolin_Studio':
        mask_labels_pbar.set_description("Processing %s" % img_name)
        if 'objects' in row['Label'].keys():
            mask_coords_dicts = row['Label']['objects'][0]['polygon']
            original_img_color = cv2.imread('F:/Invisible Man/Images/Liu_Bolin_Studio/'+img_name)
            original_img = cv2.imread('F:/Invisible Man/Images/Liu_Bolin_Studio/'+img_name, 0)
            bg = np.zeros_like(original_img).astype(np.float32)
            mask_coords = []
            for mask_coords_dict in mask_coords_dicts:
                mask_coords.append([mask_coords_dict['x'], mask_coords_dict['y']])
            contour = np.array(mask_coords, dtype=np.int32)
            # break
            bbox = cv2.boundingRect(contour)
            bounding_boxes[img_name] = bbox
            mask = cv2.fillPoly(bg, pts=[contour], color=255)
            mask = cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), 255, 3)
            cv2.imwrite('bounding_boxes/'+img_name, mask)
            # cv2.imwrite('Images/Studio_Filtered/'+img_name, original_img_color)


with open('mask_bounding_box.json', 'w') as fp:
    json.dump(bounding_boxes, fp)
