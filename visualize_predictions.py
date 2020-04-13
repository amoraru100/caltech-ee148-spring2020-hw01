import os
import numpy as np
import json
from PIL import Image

def add_bounding_boxes(I, bounding_boxes):
    # set the color of the boxes [R,G,B]
    box_color = [0,0,255]

    for box in bounding_boxes:
        tl_row = box[0]
        tl_column = box[1]
        br_row = box[2]
        br_column = box[3]

        # draw the top of the box
        I[tl_row,tl_column:br_column] = box_color

        # draw the left side of the box
        I[tl_row:br_row,tl_column] = box_color

        # draw the right side of the box
        I[tl_row:br_row,br_column] = box_color

        # draw the bottom of the box
        I[br_row,tl_column:br_column] = box_color

    return I

# set the current path
path = os.getcwd()

# set the path to the downloaded data: 
data_path = r'C:\Users\amora\Documents\Caltech\EE 148\HW1\data\RedLights2011_Medium'

# set the path to the predictions: 
preds_path = path + '\\hw01_preds' 

# set the path to the prediction visualizations:
preds_visuals_path = r'C:\Users\amora\Documents\Caltech\EE 148\HW1\data\hw01_preds_visuals'

with open(preds_path + '\\preds.json') as f:
    preds = json.load(f)

file_names = sorted(preds.keys())

for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.array(I)
    
    # add the bounding boxes
    I = add_bounding_boxes(I, preds[file_names[i]])

    image = Image.fromarray(I)
    image.save(preds_visuals_path + file_names[i])
