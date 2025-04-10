import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from IPython.display import HTML
import pandas as pd

import os
import random as r
import sys
import argparse
import csv
from datetime import datetime

sys.path.append(os.path.abspath('../blaze_common/'))
from blazedetector import BlazeDetector
from blazelandmark import BlazeLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, PoseLandmark
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS
from meanerror import MeanError


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--debug'   , default=False, action='store_true', help="Enable Debug mode. Default is off")
ap.add_argument('-a', '--an'      , default=False, action='store_true', help="Export Annotated Images. Default is off")
ap.add_argument('-v', '--view'    , default=False, action='store_true', help="View Annotated Images. Default is off")
ap.add_argument('-r', '--rd'      , default=False, action='store_true', help="Export Representative Dataset. Default is off")
ap.add_argument('-t', '--thresh'  , type=float, default=0.1, help="Error threshold at which to save images to representative dataset. Default is 0.10 (90% accuracy)")

args = ap.parse_args()  

# if args.thresh != 0.1 and not args.rd:
#     raise argparse.ArgumentError(None, "Setting a threshold requires dataset export to be enabled")

args = ap.parse_args()  

bVerbose = args.debug

#text overlay parameters
scale = 0.5
text_fontType = cv2.FONT_HERSHEY_SIMPLEX
text_fontSize = 0.75*scale
text_color    = (0,0,255)
text_lineSize = max( 1, int(1*scale) )
text_lineType = cv2.LINE_AA

# Initializing mediapipe models
blaze_detector_type = "blazepose"
blaze_landmark_type = "blazeposelandmark"
blaze_title = "BlazePoseLandmark"
default_detector_model='models/pose_detection.tflite'
default_landmark_model='models/pose_landmark_full.tflite'
# default_detector_model='models/pose_detector_28april2023.tflite'
# default_landmark_model='models/pose_landmarks_detector_28april2023.tflite'

### Lite models ###
# default_detector_model='models/pose_detector_lite_27April2023.tflite'
# default_landmark_model='models/pose_landmarks_detector_lite_27April2023.tflite'

##NXP Models ===
# default_detector_model = "models/pose_detection_quant_imx.tflite"
# default_landmark_model = "models/pose_landmark_lite_quant_imx.tflite"

## Heavy Models ##
# default_detector_model='models/pose_detector_heavy_28April2023.tflite'
#default_landmark_model='models/pose_landmarks_detector_heavy_28April2023.tflite'
# default_landmark_model='models/pose_landmark_heavy.tflite'

#### Quantized models ####
## NEWLY ADDED 0.1 QUANTIZED MODELS ##
# default_landmark_model='models/pose_landmark_yoga_0.1_quant.tflite'
# default_landmark_model='models/pose_landmark_yoga_0.1_quant_26.tflite'
#    default_detector_model='models/pose_detection_quant_floatinputs.tflite'
# default_landmark_model='models/pose_landmark_full_quant_floatinputs.tflite'

blaze_detector = BlazeDetector(blaze_detector_type)
blaze_detector.load_model(default_detector_model)

blaze_landmark = BlazeLandmark(blaze_landmark_type)
blaze_landmark.load_model(default_landmark_model)
print("================================================================")
print("MediaPipe Models Initialized")
print("================================================================")


# Base paths
results_base_path = '../../Yoga_Poses-Dataset/Results/'
train_base_path = '../../Yoga_Poses-Dataset/TRAIN/'
my_results_base_path = 'Results/'
landmark_images_base_path = 'Results/Landmark_Images/'
represetative_dataset_base_path = 'Results/Representative_Dataset/'
# results file path
results_csv_path = os.path.join(my_results_base_path, "model_comparison_results.csv")
# print(f"Checking directory creation:")
# print(f"Visualizations root exists: {os.path.exists(landmark_images_base_path)}")

# Create My Results directory for saving CSV if it doesn't exist
os.makedirs(my_results_base_path, exist_ok=True)

# Get all pose folders in TRAIN directory
pose_folders = [f for f in os.listdir(train_base_path) 
                if os.path.isdir(os.path.join(train_base_path, f))]

total_saved_images = 0
valid_pose_count = 0  # Count of poses with valid error calculations
total_count = 1 # Initialize because first pass is 0
total_error = 0
per_row_total_error = 0
per_row_total_error_minus_vis = 0
per_row_count = 1 

for pose_name in pose_folders:
    print(f"\nProcessing pose: {pose_name}")

    # Initialize DataFrame for this pose
    data = []
    points = PoseLandmark
    for p in points:
        x = str(p)[13:]
        data.append(x + "_x")
        data.append(x + "_y")
        data.append(x + "_z")
        data.append(x + "_vis")
    pose_df = pd.DataFrame(columns=data)
    
    # Path to current pose's images
    image_folder = os.path.join(train_base_path, pose_name, 'Images/')

    # Create output directories
    pose_vis_path = os.path.join(landmark_images_base_path, pose_name)
    os.makedirs(pose_vis_path, exist_ok=True)
    rep_dataset_path = os.path.join(represetative_dataset_base_path, pose_name)
    os.makedirs(rep_dataset_path, exist_ok=True)
    saved_images = 0
    count = 0

    #print("[INFO: initialized dataframe and set output directories]")
    # Process each image in the folder
    for img_file in os.listdir(image_folder):
        
        #print(f"Processing image: {img_file} in image folder: {image_folder}")
        try:
            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_file}")
                continue
            
            img_with_landmarks = img.copy()

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img1, scale1, pad1 = blaze_detector.resize_pad(imgRGB)
            
            # Run detection
            normalized_detections = blaze_detector.predict_on_image(img1)
            
            if len(normalized_detections) > 0:
                temp = []
                detections = blaze_detector.denormalize_detections(normalized_detections, scale1, pad1)

                xc, yc, scale, theta = blaze_detector.detection2roi(detections)
                roi_img, roi_affine, roi_box = blaze_landmark.extract_roi(imgRGB, xc, yc, theta, scale)

                flags, normalized_landmarks = blaze_landmark.predict(roi_img)
                
                # Store landmarks in DataFrame
                landmarks_copy = normalized_landmarks.copy()
                landmarks_list = landmarks_copy.tolist()
                # Safely process landmarks
                if not landmarks_list or len(landmarks_list[0]) < len(points):
                    print(f"Insufficient landmarks detected in {img_file}")
                    continue
                
                for i, j in zip(points, landmarks_list[0]):
                    temp.extend([j[0], j[1], j[2], j[3]])
                
                pose_df.loc[count] = temp

                landmarks = blaze_landmark.denormalize_landmarks(normalized_landmarks, roi_affine)
                
                # Visualization (optional)
                
                for i in range(len(flags)):
                    landmark, flag = landmarks[i], flags[i]
                    #if True: #flag>.5:
                    if landmarks.shape[1] > 33:
                        draw_landmarks(img_with_landmarks, landmark[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                    else:
                        draw_landmarks(img_with_landmarks, landmark[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)             
                cv2.putText(img_with_landmarks, default_detector_model, (10,10),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)
                cv2.putText(img_with_landmarks, default_landmark_model, (10,20),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)
                #draw_roi(img_with_landmarks,roi_box)

                if args.an == True:
                    base_name = os.path.splitext(img_file)[0]
                    #print(base_name)
                    image_path = os.path.join(pose_vis_path, f"{base_name}_annotated.jpg")
                    cv2.imwrite(
                        image_path,
                        img_with_landmarks
                    )

                if args.view == True:
                    cv2.imshow(f"Landmarks (right) - {img}", img_with_landmarks)
                    cv2.waitKey(0)   # Wait indefinitely for a keypress

                    #add ability to quit with 'q' key
                    if key == ord('q'):
                        break

                if count >= 0:
                    error_calculator = MeanError(results_base_path, pose_name, pose_df)
                    #print("[INFO]: Initialized MeanError Class")
                    suffix_errors = error_calculator.calculate_mean_error_per_suffix_per_row(count)
                    # print(f"\n[INFO]: Mean Error for image {count} in {pose_name} folder")
                    # print("   Mean Error for X:", {suffix_errors[0]})
                    # print("   Mean Error for Y:", {suffix_errors[1]})
                    # print("   Mean Error for Z:", {suffix_errors[2]})
                    # print("   Mean Error for Visibility:", {suffix_errors[3]})
                    # print("   Total Mean Error (minus vis):", {suffix_errors[4]})

                    per_row_total_error+= suffix_errors[5] # Total error
                    per_row_total_error_minus_vis+= suffix_errors[4] # Total error minus vis
                    per_row_count += 1 # Increment count for mean error calculation
                    #print("Per row mean error", {suffix_errors[5]}) 

                if args.rd == True:    
                    ###  IF MEAN ERROR < THRESHOLD, EXPORT ROI IMAGES TO USE FOR QUANTIZATION
                    if suffix_errors[4] < error_threshold:
                        saved_images += 1
                        roi_img_squeezed = (np.squeeze(roi_img, axis=0))  # Now shape (256, 256, 3) and values in range [0, 255]

                        # Step 1: Ensure values are in [0, 255]
                        if roi_img_squeezed.dtype == np.float32:
                            roi_img_squeezed = (roi_img_squeezed * 255).astype(np.uint8)  # Float [0,1] → Uint8 [0,255]
                        elif roi_img_squeezed.dtype == np.float64:
                            roi_img_squeezed = (roi_img_squeezed * 255).astype(np.uint8)  # Same for float64

                        roi_img_bgr = cv2.cvtColor(roi_img_squeezed, cv2.COLOR_RGB2BGR)
                        # Export the ROI image for quantization. Ideally it should be 224x224? Or should it be larger?
                        roi_img_path = os.path.join(represetative_dataset_base_path, f"{img_file}_roi.jpg") # chane to rep_dataset_path to organize by pose
                        cv2.imwrite(roi_img_path, roi_img_bgr)
                        # cv2.imshow(f"Landmarks (right) - {img}", roi_img_bgr)
                        # cv2.waitKey(0)   # Wait indefinitely for a keypress
                        # if key == ord('q'):
                        #     break
                        print(f"Exported ROI image for quantization: {roi_img_path}") 

                
                count += 1 # store value after updating the dataframe. In the spreadsheet, row 1 gets count=0


        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    number_of_images = len(os.listdir(image_folder))
    # Calculate pose error only if we have processed all images
    if count == number_of_images and number_of_images > 0:
        try:
            total_error_calculator = MeanError(results_base_path, pose_name, pose_df)
            pose_error = total_error_calculator.calculate_mean_error()
            if not np.isnan(pose_error):
                total_error += pose_error
                valid_pose_count += 1
                print(f"Mean error for blaze's {pose_name}: {pose_error:.4f}")
            else:
                print(f"Warning: Could not calculate valid error for {pose_name}.\n This is likely due to blaze's detections > groundtruth detections")
        except Exception as e:
            print(f"Error calculating pose error for {pose_name}: {str(e)}")
    
    # print(pose_df.shape)
    # print(number_of_images)
    # if (len(pose_df)) == number_of_images:
    #     total_error_calculator = MeanError(results_base_path, pose_name, pose_df)
    #     pose_error = total_error_calculator.calculate_mean_error()
    #     print(f"Pose error: {pose_error:.4f}")
    #     total_error += pose_error

    # Save results for this pose
    output_path = os.path.join(my_results_base_path, f"Dataset_{pose_name}.csv")
    pose_df.to_csv(output_path)
    print(f"Saved results for {pose_name} with {count} processed images to {output_path}")
    if args.rd:
        print("saved " + str(saved_images) + " out of " + str(len(os.listdir(image_folder))) + " images")

    total_saved_images += saved_images
    total_count += count

print("Total number of images processed: ", total_count)
cv2.destroyAllWindows()
# print("\nProcessing complete for all poses!")
# Calculate final mean error
total_folders = len(pose_folders)
# print("Total number of pose folders: ", total_folders)
# print("Total number of valid pose folders: ", valid_pose_count)
if valid_pose_count > 0:
    total_mean_error = total_error /valid_pose_count
else:
    total_mean_error = float('nan')

original_total_images = 0
for folder in pose_folders:
    original_total_images+= len(os.listdir(os.path.join(train_base_path, folder, 'Images/')))

#total_mean_error = total_error / len(pose_folders)
print("================================================================")
print("Final Results for detection model " + default_detector_model + " and landmark model " + default_landmark_model)

print("Total Mean Error: ", total_mean_error) 
print("Total per row mean error: ", per_row_total_error/per_row_count) 
print("Total per row mean error minus vis for detection model: ", per_row_total_error_minus_vis/per_row_count) 

print(f"Total processed images: {total_count}")
print("Original total images: ", original_total_images)
if args.rd:
    print("saved " + str(total_saved_images) + " out of " + str(total_count) + " images to dataset")
# print("per row count: ", per_row_count)
print("================================================================")

# After your final print statements, add this:
results_data = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'detection_model': os.path.basename(default_detector_model),
    'landmark_model': os.path.basename(default_landmark_model),
    'total_mean_error': total_mean_error,
    'per_row_mean_error': per_row_total_error/per_row_count if per_row_count > 0 else float('nan'),
    'per_row_mean_error_minus_vis': per_row_total_error_minus_vis/per_row_count if per_row_count > 0 else float('nan'),
    'total_images_processed': total_count,
    'representative_images_saved': total_saved_images
}

# Write/append to CSV
file_exists = os.path.exists(results_csv_path)
with open(results_csv_path, 'a', newline='') as csvfile:
    fieldnames = results_data.keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if not file_exists:
        writer.writeheader()  # Write header only if file doesn't exist
    writer.writerow(results_data)

print(f"\nResults appended to: {results_csv_path}")
