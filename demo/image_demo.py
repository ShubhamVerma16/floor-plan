# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import torch
import numpy as np
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot

#config = "configs/point_rend/pointrend_r101_4xb2-80k_floorplan-512x1024.py"
config = "work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes_floorplan-512x1024/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes_floorplan-512x1024.py"
checkpoint = "work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes_floorplan-512x1024/iter_90000.pth"
#checkpoint = "work_dirs/pointrend_r101_4xb2-80k_floorplan-512x1024_one_class_r4/iter_80000.pth"
#checkpoint = "work_dirs/pointrend_r101_4xb2-80k_floorplan-512x1024/iter_36000.pth"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_model(config, checkpoint, device=device)
if device == 'cpu':
        model = revert_sync_batchnorm(model)

# Function to reduce contour points
def reduce_contour_points(contours, epsilon_ratio=0.02):
    reduced_contours = []
    for contour in contours:
        # Calculate epsilon as a fraction of the contour's perimeter
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, epsilon, True)
        reduced_contours.append(approx)
    return reduced_contours

def dense_noisy_area(contour, density_threshold = 0.01):
    area = cv2.contourArea(contour)
    if area <= 0:
        return True

    # Calculate the number of points in the contour
    num_points = len(contour)

    # Calculate point density
    density = num_points / area
    print("Contour: density points: ", density)

    # Check if density exceeds the threshold
    if density > density_threshold:
        return True

def valid_reduced_contour(contour):
    if dense_noisy_area(contour): #or far_noisy_area(contour) or long_noisy_area(contour):
        return False
    else:
        return True

def inference(img_nm, out_nm):
    img = f"data/floorplan_point_rend/eval/300-499/{img_nm}"
    
    out_file = f"outputs/api_res/{out_nm}.png"
    opacity = 1
    with_labels = False
    title = "result"
    #parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    #parser.add_argument('config', help='Config file')
    #parser.add_argument('checkpoint', help='Checkpoint file')
    #parser.add_argument('--out-file', default=None, help='Path to output file')
    #parser.add_argument(
    #    '--device', default='cuda:0', help='Device used for inference')
    #parser.add_argument(
    #    '--opacity',
    #    type=float,
    #    default=0.5,
    #    help='Opacity of painted segmentation map. In (0, 1] range.')
    # parser.add_argument(
    #     '--with-labels',
    #     action='store_true',
    #     default=False,
    #     help='Whether to display the class labels.')
    # parser.add_argument(
    #     '--title', default='result', help='The image identifier.')
    #args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    # test a single image
    #image = cv2.imread(img)
    result = inference_model(model, img)
    #print("result: ", type(result), result)
    # show the results
    save_dir = "outputs/api_res/"
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
    #cv2.imwrite(save_dir + out_nm + "_bound.png", seg_img_rgb)
    #print("Completed processing")

    #print(contours)

    #generate json
    output = []
    result = {}
    result["image_nm"] = img.split('/')[-1]
    result["annotations"] = []
    # Draw the point (small circle)
    point_color = (255, 0, 0)  # Red color in BGR
    point_radius = 10  # Radius of the point
    point_thickness = -1  # Fill the circle

    new_result = {}
    new_result["image_nm"] = img.split('/')[-1]
    new_result["annotations"] = []

    for idx, cont in enumerate(contours):
        #print("Cont: ", cont)
        cont_inf = {}
        M = cv2.moments(cont)
        cX, cY = 0, 0
        # Calculate the x and y coordinates of the center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        """cont_x, cont_y = [pt[0][0] for pt in cont], [pt[0][1] for pt in cont]
        min_x, max_x, min_y, max_y = min(cont_x), max(cont_x), min(cont_y), max(cont_y)
        cv2.circle(seg_img_rgb, (min_x, min_y), point_radius, point_color, point_thickness)
        cv2.circle(seg_img_rgb, (min_x, min_y), 15, point_color, 2)

        cv2.circle(seg_img_rgb, (min_x, max_y), point_radius, point_color, point_thickness)
        cv2.circle(seg_img_rgb, (min_x, max_y), 15, point_color, 2)

        cv2.circle(seg_img_rgb, (max_x, min_y), point_radius, point_color, point_thickness)
        cv2.circle(seg_img_rgb, (max_x, min_y), 15, point_color, 2)

        cv2.circle(seg_img_rgb, (max_x, max_y), point_radius, point_color, point_thickness)
        cv2.circle(seg_img_rgb, (max_x, max_y), 15, point_color, 2)"""

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
    

    ############### contour reduction plotting ##################
    # Reduce contour points
    epsilon_ratio = 0.003  # Adjust this value as needed
    reduced_contours = reduce_contour_points(contours, epsilon_ratio)
    new_reduced_conts = []
    height, width, _ = seg_img_rgb.shape

    for contour in reduced_contours:
        if valid_reduced_contour(contour):
            red_cont_inf = {}
            red_cont_inf["area_type"] = "indoor"
            red_cont_inf["points"] = contour.tolist()
            red_cont_inf["area"] = cv2.contourArea(contour)
            red_cont_inf["reduced_points"] = len(contour.tolist())
            #red_cont_inf["confidence"] = 0.64
            new_result["annotations"].append(red_cont_inf)

            cv2.drawContours(seg_img_rgb, [contour], -1, (2, 48, 32), 2)  # Reduced contours in red
            for point in contour:
                cv2.circle(seg_img_rgb, tuple(point[0]), 4, (255, 0, 0), -1)  # Mark reduced points in blue

        else:
            fill_color = (255, 255, 255)
            xmin, ymin, w, h = cv2.boundingRect(contour)
            for point in contour:
                x, y = point[0]
                dx = -3 if x < (xmin + (w//2)) else 3
                dy = -3 if y < (ymin + (h//2)) else 3
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    b, g, r = seg_img_rgb[ny, nx]
                    if tuple((b,g,r)) == tuple((0, 245, 0)):
                        #print("Fill color green")
                        fill_color = (0, 245, 0)
            cv2.drawContours(seg_img_rgb, [contour], -1, fill_color, -1)  # noisy contours in white

    output.append(new_result)
    cv2.imwrite(save_dir + out_nm + "_bound.png", seg_img_rgb)

    return output
    

if __name__ == '__main__':
    main()