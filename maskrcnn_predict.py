# USAGE
# python maskrcnn_predict.py --weights mask_rcnn_coco.h5 --labels coco_labels.txt --image images/30th_birthday.jpg

# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-w", "--weights", required=True,help="path to Mask R-CNN model weights pre-trained on COCO")
#ap.add_argument("-l", "--labels", required=True,help="path to class labels file")
ap.add_argument("-i", "--image", required=True,help="path to input image to apply Mask R-CNN to")
args = vars(ap.parse_args())

# load the class label names from disk, one label per line
CLASS_NAMES = ["background","crack","dent","glass","out","scratch"]

# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

class SimpleConfig(Config):
     # give the configuration a recognizable name
     NAME = "coco_inference"

     # set the number of GPUs to use along with the number of images
     # per GPU
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1

     # number of classes (we would normally add +1 for the background
     # but the background class is *already* included in the class
     # names)
     NUM_CLASSES = len(CLASS_NAMES)

# initialize the inference configuration
config = SimpleConfig()

# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
crack_weights = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
dent_weights = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
glass_weights = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
out_weights = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
scratch_weights = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
crack_weights.load_weights("weights/crack.h5", by_name=True,exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
dent_weights.load_weights("weights/dent.h5", by_name=True,exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
glass_weights.load_weights("weights/glass.h5", by_name=True,exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
out_weights.load_weights("weights/out.h5", by_name=True,exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
scratch_weights.load_weights("weights/scratches.h5", by_name=True,exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# load the input image, convert it from BGR to RGB channel
# ordering, and resize the image

#image is used for testing
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)

#img is used for  applying all the masks
img = cv2.imread(args["image"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = imutils.resize(img, width=512)

# perform a forward pass of the network to obtain the results
print("[INFO] making predictions with Mask R-CNN...")
crack_results = crack_weights.detect([image], verbose=1)[0]
dent_results = dent_weights.detect([image], verbose=1)[0]
glass_results = glass_weights.detect([image], verbose=1)[0]
out_results = out_weights.detect([image], verbose=1)[0]
scratch_results = scratch_weights.detect([image], verbose=1)[0]

# loop over of the detected object's bounding boxes and masks

#generate masks for dents
for i in range(0, dent_results["rois"].shape[0]):
      #extract the class ID and mask for the current detection, then
     #grab the color to visualize the mask (in BGR format)
     classID = dent_results["class_ids"][i]
     dent_mask = dent_results["masks"][:, :, i]
     color = COLORS[classID][::-1]

     if dent_results["scores"][i] > 0.9:
          # visualize the pixel-wise mask of the object
          img = visualize.apply_mask(img, dent_mask, color, alpha=0.5)

#generate masks for cracks
for i in range(0, crack_results["rois"].shape[0]):
     # extract the class ID and mask for the current detection, then
     # grab the color to visualize the mask (in BGR format)
     classID = crack_results["class_ids"][i]
     crack_mask = crack_results["masks"][:, :, i]
     color = COLORS[classID][::-1]

     if crack_results["scores"][i] > 0.9:
          # visualize the pixel-wise mask of the object
          img = visualize.apply_mask(img, crack_mask, color, alpha=0.5)

#generate masks for glass
for i in range(0, glass_results["rois"].shape[0]):
     # extract the class ID and mask for the current detection, then
     # grab the color to visualize the mask (in BGR format)
     classID = glass_results["class_ids"][i]
     glass_mask = glass_results["masks"][:, :, i]
     color = COLORS[classID][::-1]

     if glass_results["scores"][i] > 0.9:
          # visualize the pixel-wise mask of the object
          img = visualize.apply_mask(img, glass_mask, color, alpha=0.5)

#generate masks for out
for i in range(0, out_results["rois"].shape[0]):
     # extract the class ID and mask for the current detection, then
     # grab the color to visualize the mask (in BGR format)
     classID = out_results["class_ids"][i]
     out_mask = out_results["masks"][:, :, i]
     color = COLORS[classID][::-1]

     if out_results["scores"][i] > 0.9:
          # visualize the pixel-wise mask of the object
          img = visualize.apply_mask(img, out_mask, color, alpha=0.5)

#generate masks for scratches
for i in range(0, scratch_results["rois"].shape[0]):
     # extract the class ID and mask for the current detection, then
     # grab the color to visualize the mask (in BGR format)
     classID = scratch_results["class_ids"][i]
     scratch_mask = scratch_results["masks"][:, :, i]
     color = COLORS[classID][::-1]

     if scratch_results["scores"][i] > 0.9:
          # visualize the pixel-wise mask of the object
          img = visualize.apply_mask(img, scratch_mask, color, alpha=0.5)
# convert the image back to BGR so we can use OpenCV's drawing
# functions
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#mask_sum_x = 0
#mask_sum_y = 0
#count = 0
#for i in range(0,r["masks"].shape[0]):
#     for j in range(0,r["masks"].shape[1]):
#          if r["masks"][i][j] == True:
#               #print("("+str(i)+","+str(j)+")")
#               mask_sum_x += i
#               mask_sum_y += j
#               count += 1
#
#mask_centroid_x = mask_sum_x / count
#mask_centroid_y = mask_sum_y / count
center = 0
# loop over the predicted scores and class labels of cracks
for i in range(0, len(crack_results["scores"])):
     # extract the bounding box information, class ID, label, predicted
     # probability, and visualization color
     (startY, startX, endY, endX) = crack_results["rois"][i]
     classID = crack_results["class_ids"][i]
     label = CLASS_NAMES[classID]
     score = crack_results["scores"][i]
     color = [int(c) for c in np.array(COLORS[classID]) * 255]
     if score > 0.9:
           center = int(( startX + endX )/2)
           # draw the bounding box, class label, and score of the object
           cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
           text = "{}: {:.3f}".format(label, score)
           y = startY - 10 if startY - 10 > 10 else startY + 10
           cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

center = 0
# loop over the predicted scores and class labels of dents
for i in range(0, len(dent_results["scores"])):
     # extract the bounding box information, class ID, label, predicted
     # probability, and visualization color
     (startY, startX, endY, endX) = dent_results["rois"][i]
     classID = dent_results["class_ids"][i]
     label = CLASS_NAMES[classID]
     score = dent_results["scores"][i]
     color = [int(c) for c in np.array(COLORS[classID]) * 255]
     if score > 0.9:
           center = int(( startX + endX )/2)
           # draw the bounding box, class label, and score of the object
           cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
           text = "{}: {:.3f}".format(label, score)
           y = startY - 10 if startY - 10 > 10 else startY + 10
           cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

center = 0
# loop over the predicted scores and class labels of glass
for i in range(0, len(glass_results["scores"])):
     # extract the bounding box information, class ID, label, predicted
     # probability, and visualization color
     (startY, startX, endY, endX) = glass_results["rois"][i]
     classID = glass_results["class_ids"][i]
     label = CLASS_NAMES[classID]
     score = glass_results["scores"][i]
     color = [int(c) for c in np.array(COLORS[classID]) * 255]
     if score > 0.9:
           center = int(( startX + endX )/2)
           # draw the bounding box, class label, and score of the object
           cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
           text = "{}: {:.3f}".format(label, score)
           y = startY - 10 if startY - 10 > 10 else startY + 10
           cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

center = 0
# loop over the predicted scores and class labels of out
for i in range(0, len(out_results["scores"])):
     # extract the bounding box information, class ID, label, predicted
     # probability, and visualization color
     (startY, startX, endY, endX) = out_results["rois"][i]
     classID = out_results["class_ids"][i]
     label = CLASS_NAMES[classID]
     score = out_results["scores"][i]
     color = [int(c) for c in np.array(COLORS[classID]) * 255]
     if score > 0.9:
           center = int(( startX + endX )/2)
           # draw the bounding box, class label, and score of the object
           cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
           text = "{}: {:.3f}".format(label, score)
           y = startY - 10 if startY - 10 > 10 else startY + 10
           cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

center = 0
# loop over the predicted scores and class labels of scratches
for i in range(0, len(scratch_results["scores"])):
     # extract the bounding box information, class ID, label, predicted
     # probability, and visualization color
     (startY, startX, endY, endX) = scratch_results["rois"][i]
     classID = scratch_results["class_ids"][i]
     label = CLASS_NAMES[classID]
     score = scratch_results["scores"][i]
     color = [int(c) for c in np.array(COLORS[classID]) * 255]
     if score > 0.9:
           center = int(( startX + endX )/2)
           # draw the bounding box, class label, and score of the object
           cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
           text = "{}: {:.3f}".format(label, score)
           y = startY - 10 if startY - 10 > 10 else startY + 10
           cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# show the output image
#cv2.imshow("Output", image)
#cv2.circle(image,(int(mask_centroid_y),int(mask_centroid_x)),5,(0,0,255), 2)
#print("("+str(int(mask_centroid_x))+","+str(int(mask_centroid_y))+")")
#if(int(mask_centroid_y)> center):
 #    print("right")
#else:
 #    print("left")
cv2.imwrite("Output.jpg", img)
cv2.waitKey()
