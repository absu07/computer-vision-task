import argparse
import cv2
#import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import time
from glob import glob

from utils import OnnxRunner

IMG_EXT = ["jpg","jpeg","png","bmp"]

INP_TYPE = {
    0:"IMG",
    1:"FOLDER"
}

def parse_input(inp:str)->str:
    """
    In order to correctly manage the 2 cases : 
    single image or folder of images in a good way, we just 
    use this small checking function. 
    No, it's not over-engineering, if we want to add video handling
    thanks to this code it's clean and easy :D

    args:
        inp (str) : input path defined as an argument of this script

    returns:
        str : data type code defined in INP_TYPE

    """

    if(os.path.isfile(inp)):
        ext = inp.split(".")[-1]
        if(ext in IMG_EXT):
            return INP_TYPE[0]
    
    return INP_TYPE[1]

def preProcessFrame(frame:np.ndarray)->np.ndarray:
    """
    Preprocess function. The model needs a certain format of image to work and this function
    is meant to make the input suits the model requirements.

    Requirements : 
        - All the checkpoints have been exported with an input size of 640x640.
        - All the checkpoints take an RGB BCHW format.
        - All the checkpoints need the input to be normalized according to the instructions.
        - All the checkpoints need the input to be a np.ndarray of dtype=float32
    
    args:
        frame (np.ndarray) : raw np.ndarray frame that should be processed

    returns:
        np.ndarray : processed frame as described in the requirements just above
    
    """

    # Resize the frame to 640x640
    frame_resized = cv2.resize(frame, (640, 640))
    
    # Convert frame color channel from BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Standardize the pixel values
    frame_standardized = (frame_rgb - 127.5) / 127.5
    
    # Change data format to BCHW (Batch, Channels, Height, Width)
    frame_bchw = np.transpose(frame_standardized, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    
    return frame_bchw

# def postProcessFrame(metadata:list, frame:np.ndarray)->np.ndarray:
#     """
#     In order to display on the image the model's result,
#     this function will draw a list of metadata on a frame and 
#     return the result.

#     args : 
#         metadata (list) : list of scaled bounding boxes [x1,y1,x2,y2,score]
#         frame (numpy.ndarray) : plain frame

#     returns :
#         frame (numpy.ndarray) : frame with bounding boxes drawn on it
#     """

#     # Make a copy of the frame to draw on
#     annotated_frame = frame.copy()

#     # Iterate over each detected object
#     for box in metadata:
#         # Extract coordinates and score
#         x1, y1, x2, y2, score = box
#         print(f"x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}, score {score}")
        
#         # Convert coordinates to integer if they are not already
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
#         print("Drawing bounding box...")
#         # Draw bounding box
#         cv2.rectangle(annotated_frame, (0, 0), (100, 100), (0, 255, 0), 2)
#         print("Bounding box drawn successfully.")
        
#         # Put score text
#         label = f"{score:.2f}"
#         cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#     cv2.imshow('Predictions', annotated_frame)
    
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     return annotated_frame

def postProcessFrame(metadata:list, frame:np.ndarray)->np.ndarray:
    """
    In order to display on the image the model's result,
    this function will draw a list of metadata on a frame and 
    return the result.

    args : 
        metadata (list) : list of scaled bounding boxes [x1,y1,x2,y2,score]
        frame (numpy.ndarray) : plain frame

    returns :
        frame (numpy.ndarray) : frame with bounding boxes drawn on it
    """

    # Get the height and width of the image
    height, width, _ = frame.shape

    # Make a copy of the frame to draw on
    annotated_frame = frame.copy()

    # Iterate over each detected object
    for box in metadata:
        # Extract coordinates and score
        x1, y1, x2, y2, score = box
        # print(f"x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}, score {score}")

        # Convert normalized coordinates to pixel values
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        # print("Drawing bounding box...")
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # print("Bounding box drawn successfully.")

        # Put score text
        label = f"{score:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # cv2.imshow('Predictions', annotated_frame)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return annotated_frame


# def ProcessFrame(model:object,frame:np.ndarray):
#     """
#     Main function that will take a raw OpenCV frame,
#     will preprocess it, will infer it using the onnx model
#     and will postprocess the result in order to draw the 
#     bounding boxes on the frame and save it.

#     args : 
#         model (OnnxRunner) : OnnxRunner that wraps an ORT object, able to be infered
#         frame (np.ndarray) : raw frame HWC

#     returns : 
#         np.ndarray : same frame as input but with bounding boxes drawn on it
#     """

#     processed_frame = preProcessFrame(frame)
#     metadata = model.run(processed_frame)
#     print(f"Metadata: {metadata}")
#     annotated_frame = postProcessFrame(metadata, frame)

#     return annotated_frame

def load_ground_truth(gt_path: str) -> list:
    """
    Load ground truth bounding boxes from a text file.

    args:
        gt_path (str): Path to the ground truth text file.

    returns:
        list: List of ground truth bounding boxes [x1, y1, x2, y2].
    """
    ground_truth = []
    with open(gt_path, 'r') as file:
        for line in file:
            #Map the ground truth co-ordinates to a list
            coords = list(map(float, line.strip().split()))
            ground_truth.append(coords)
    return ground_truth

def iou(box1: list, box2: list) -> float:
    """
    args:

    Calculate Intersection over Union (IoU) for two bounding boxes.

        box1 (list): First bounding box [x1, y1, x2, y2].
        box2 (list): Second bounding box [x1, y1, x2, y2].

    returns:
        float: IoU score.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou_score = inter_area / float(box1_area + box2_area - inter_area)
    return iou_score

def ProcessFrame(model: object, frame: np.ndarray, ground_truth: list, iou_threshold: float) -> np.ndarray:
    processed_frame = preProcessFrame(frame)
    metadata = model.run(processed_frame)
    # print(f"Metadata: {metadata}")
    
    height, width, _ = frame.shape
    
    correct_predictions = 0
    for gt_box in ground_truth:
        for pred_box in metadata:
            pred_coords = pred_box[:4]  # Extract only the coordinates
            # print(f"pred_coords: {pred_coords}")
            # print(f"gt_box: {gt_box}")
            # Convert normalized predicted coordinates to pixel values
            pred_coords[0] = pred_coords[0] * width  # x1
            pred_coords[1] = pred_coords[1] * height  # y1
            pred_coords[2] = pred_coords[2] * width  # x2
            pred_coords[3] = pred_coords[3] * height  # y2
            
            iou_score = iou(gt_box, pred_coords)
            # print(f"iou_score: {iou_score}")
            if iou_score >= iou_threshold:
                correct_predictions += 1
                break

    print(f"Correct Predictions: {correct_predictions}/{len(ground_truth)}")

    accuracy = correct_predictions / len(ground_truth)
    annotated_frame = postProcessFrame(metadata, frame)
    
    return annotated_frame, accuracy

def getOutputPath(input:str, output:str)->str:
    """
    Function that takes input file path & output folder path and 
    returns the output file path.

    args:
        input (str) : input file path (/path/to/image.jpg)
        output (str) : output folder path (/path/to/output/)
    
    returns: 
        str : output file path (/path/to/output/image_output.jpg)
    """

    # file_name, extension = input.split("\\")[-1].split(".")

    # # import pdb;pdb.set_trace()
    # # output = "D:\Personal\Interviews\ML_SmartCowTakeHomeAssignment_2024\ML_SmartCowTakeHomeAssignment\Take_Home_Computer_Vision\computer_vision\computer-vision-task\results"
    # # temp = os.path.join(output_x,file_name+"_output."+extension)
    
    # file_name = '\\'.join(file_name.split('\\')[:-2])
    # # import pdb;pdb.set_trace()
    # temp_file = input.split("/")[-1].split(".")[0].split('\\')[-1]
    # test_path =  os.path.join(output,temp_file+"_output."+extension)
    # # import pdb;pdb.set_trace()
    # return os.path.join(output,temp_file+"_output."+extension)
    
    file_name, extension = input.split("/")[-1].split(".")
    return os.path.join(output,file_name+"_output."+extension)

def evaluate_models(onnx_files: list, img_list: list, gt_list: list, output_folder: str, iou_threshold: float, conf_threshold: float):
    best_accuracy = 0
    best_model = None

    for onnx_file in onnx_files:
        print(f"Evaluating {onnx_file}...")
        model = OnnxRunner(onnx_file, nms_thresh=iou_threshold, conf_thresh=conf_threshold)
        total_accuracy = 0

        for idx, img_path in enumerate(img_list):
            ground_truth = load_ground_truth(gt_list[idx])
            image = cv2.imread(img_path)
            output_path = getOutputPath(img_path, "D:\Personal\Interviews\ML_SmartCowTakeHomeAssignment_2024\ML_SmartCowTakeHomeAssignment\Take_Home_Computer_Vision\computer_vision\computer-vision-task\results")
            annotated_frame, accuracy = ProcessFrame(model, image, ground_truth, iou_threshold)
            total_accuracy += accuracy
            cv2.imwrite(output_path, annotated_frame)

        avg_accuracy = total_accuracy / len(img_list)
        print(f"Model: {onnx_file}, Average Accuracy: {avg_accuracy:.4f}")

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_model = onnx_file

    print(f"Best Model: {best_model} with Average Accuracy: {best_accuracy:.4f}")
    return best_model

def main(args:object)->None:
    """
    Main function that will parse the input in order to manage
    the 2 cases : single image or folder of images.

    args : 
        - input (str)
        - onnx (str)
        - output (str)
        - iou_threshold (float)
        - conf_threshold (float)
    """

    # model = OnnxRunner(args.onnx, nms_thresh=args.iou_threshold, conf_thresh=args.conf_threshold)

    input_type = parse_input(args.input)  
    onnx_files = sorted(glob(os.path.join(args.onnx, "*.onnx")))

    if(input_type=="IMG"):
        # Load ground truth data
        # ground_truth = load_ground_truth(args.gt)
        # print(f"[INFO] Processing 1 image : {args.input}")

        # TIME_FLAG = time.time()

        # image = cv2.imread(args.input)
        # output_path = getOutputPath(args.input, args.output)
        # annotated_frame, accuracy = ProcessFrame(model, image, ground_truth, args.iou_threshold)

        # best_model = evaluate_models(onnx_files, image, ground_truth, args.output, args.iou_threshold, args.conf_threshold)
        # # print(f"[INFO] Processed image in {time.time()-TIME_FLAG}s")

        # cv2.imwrite(output_path, annotated_frame)
        img_list = [args.input]
        gt_list = [args.gt]

        # return

    if(input_type=="FOLDER"):

        img_list = sorted(glob(os.path.join(args.input,"*")))
        gt_list = sorted(glob(os.path.join(args.gt, "*")))

        # print(f"[INFO] Processing a folder containing {len(img_list)} images.")

        # TIME_FLAG = time.time()
        # for idx,img_path in enumerate(img_list):
        #     # print(f"{idx}  image {img_path}")
        #     # print(f"ground turth path {gt_list[idx]}")
        #     # import pdb;pdb.set_trace()

        # #     print(f"image_path {img_path}")
        #     ground_truth = load_ground_truth(gt_list[idx])
            
        #     image = cv2.imread(img_path)
        #     output_path = getOutputPath(img_path, args.output)
        #     # print(f"output saved here {output_path}")
        #     # print(f"output path: {args.output}")
        #     # import pdb;pdb.set_trace()
        #     annotated_frame, accuracy = ProcessFrame(model, image, ground_truth, args.iou_threshold)
        #     best_model = evaluate_models(onnx_files, image, ground_truth, args.output, args.iou_threshold, args.conf_threshold)
            
        #     cv2.imwrite(output_path, annotated_frame)

        # print(f"[INFO] Processed {len(img_list)} images in {time.time()-TIME_FLAG}s. Avg : {(time.time()-TIME_FLAG)/len(img_list)}s")

        # return
    best_model = evaluate_models(onnx_files, img_list, gt_list, args.output, args.iou_threshold, args.conf_threshold)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Computer Vision Assignement')

    parser.add_argument('--input','-i', type=str, required=True, help="Data input path. Can be an image or a folder of images.")
    parser.add_argument('--onnx','-o', type=str, required=True, help="Onnx model path.")
    parser.add_argument('--output','-p', type=str, required=False, help="Output folder path. The infered and annotated data will be written in this folder. Default : results.", default="results")
    parser.add_argument('--iou_threshold','-u', type=float, required=False, help="IOU threshold used in NMS. Default : 0.4", default=0.1)
    parser.add_argument('--conf_threshold','-c', type=float, required=False, help="Confidence score threshold. Default : 0.5", default=0.4)
    parser.add_argument('--gt', '-g', type=str, required=True, help="Path to the ground truth text file.")


    args = parser.parse_args()

    main(args)
