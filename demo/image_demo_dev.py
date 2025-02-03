# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import torch
import numpy as np
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot

config = "configs/point_rend/pointrend_r101_4xb2-80k_floorplan-512x1024.py"
checkpoint = "work_dirs/pointrend_r101_4xb2-80k_floorplan-512x1024/iter_56000.pth"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_model(config, checkpoint, device=device)
if device == 'cpu':
        model = revert_sync_batchnorm(model)
out_file = "output.png"
opacity = 1
with_labels = False
title = "result"
img = "data/floorplan_point_rend/eval/300-499/newhouse317 Recovered NA.png"


def main():
    """img = f"data/floorplan_point_rend/eval/300-499/{img_nm}"
    
    out_file = f"outputs/api_res/{out_nm}.png"
    opacity = 1
     """
    #parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    #parser.add_argument('config', help='Config file')
    #parser.add_argument('checkpoint', help='Checkpoint file')
    #parser.add_argument('--out-file', default=None, help='Path to output file')
    #parser.add_argument(
    #   '--device', default='cuda:0', help='Device used for inference')
    #parser.add_argument(
    #   '--opacity',
    #   type=float,
    #   default=0.5,
    #   help='Opacity of painted segmentation map. In (0, 1] range.')
    #parser.add_argument(
    #    '--with-labels',
    #    action='store_true',
    #    default=False,
    #    help='Whether to display the class labels.')
    #parser.add_argument(
    #    '--title', default='result', help='The image identifier.')
    #args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    # test a single image
    image = cv2.imread(img)
    print("Image shape size: ", image.shape)
    result = inference_model(model, img)
    #print("result: ", type(result), result)
    # show the results
    save_dir = "outputs/dev_res/"
    seg_img = show_result_pyplot(
        model,
        img,
        result,
        title=title,
        save_dir=save_dir,
        opacity=opacity,
        with_labels=with_labels,
        draw_gt=False,
        show=False if out_file is not None else True,
        out_file=out_file)

    #print(type(seg_img), seg_img.shape)
    #print(seg_img)

    white = np.array([255, 255, 255])
    green = np.array([0, 245, 0])
    blue = np.array([0, 0, 255])
    seg_img_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    #blue = np.array([255, 0, 0])
    seg_img[np.all(seg_img == white, axis=-1)] = [0, 0, 0]
    #cv2.imwrite(save_dir + "contrast_image.png", seg_img)
    #seg_map = cv2.imread(save_dir + out_file)
    ##print("Read img successfully", shape(seg_map))

    seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    # Optionally, threshold the grayscale image if needed
    #_, seg_img_binary = cv2.threshold(seg_img_gray, 0, 255, cv2.THRESH_BINARY)
    #seg_img_binary = cv2.bitwise_not(seg_img_binary)
    cv2.imwrite(save_dir + "gray_img.png", seg_img_gray)

    contours, hier = cv2.findContours(seg_img_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_img_rgb, contours, -1, (0, 0, 255), 5)
    cv2.imwrite(save_dir + "seg_output.png", seg_img_rgb)
    #print("Completed processing")

    #print(contours)

    #generate json
    output = []
    result = {}
    result["image_nm"] = img.split('/')[-1]
    result["annotations"] = []
    for idx, cont in enumerate(contours):
        #print("Cont: ", cont)
        cont_inf = {}
        M = cv2.moments(cont)
        # Calculate the x and y coordinates of the center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Centroid of contour {idx}: ({cY}, {cX})")
            print("center co-ord: ", seg_img[cY][cX])

        #cont_x, cont_y = [pt[0][0] for pt in cont], [pt[0][1] for pt in cont]
        #min_x, max_x, min_y, max_y = min(cont_x), max(cont_x), min(cont_y), max(cont_y)
        #print("Pixel val: ", seg_img[(min_y+max_y)//2][(min_x+max_x)//2])
        if np.all(seg_img[cY][cX] == green):
            cont_inf["area_type"] = "indoor"
            cont_inf["points"] = cont.tolist()
            cont_inf["area"] = cv2.contourArea(cont)
            cont_inf["confidence"] = 0.64
        elif np.all(seg_img[cY][cX] == blue):
            cont_inf["area_type"] = "outdoor"
            cont_inf["points"] = cont.tolist()
            cont_inf["area"] = cv2.contourArea(cont)
            cont_inf["confidence"] = 0.64
        
        result["annotations"].append(cont_inf)
    
    output.append(result)

    print(output)
    

if __name__ == '__main__':
    main()
