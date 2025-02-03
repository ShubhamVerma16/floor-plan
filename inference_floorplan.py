import os
import cv2
import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot

img_dir = "data/floorplan_point_rend/eval/"
config = "configs/point_rend/pointrend_r101_4xb2-80k_floorplan-512x1024.py"
checkpoint = "work_dirs/pointrend_r101_4xb2-80k_floorplan-512x1024/iter_48000.pth"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cfg_options = None

model = init_model(config, checkpoint, device, cfg_options)
print("Loaded model successfully", model)

print("\n#############\n", model.cfg, "model type", type(model))

for img_nm in os.listdir(img_dir):
    img_pth = img_dir + img_nm #os.path.join(img_dir, img_nm)
    img = cv2.imread(img_pth)
    result = inference_model(model, img)
    print("Model inferred successfully: ", type(result), len(result), "\n", result)

    seg_img = show_result_pyplot(model, 
                                img_pth, 
                                result,
                                save_dir = "outputs/api_res",
                                out_file= "outputs/api_res/"+img_nm.split('.')[0]+".txt")