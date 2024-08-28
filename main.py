import argparse
import cv2
import os
import numpy as np
import sys
import time
import logging
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
    
    return annotated_frame

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
    """
    Process a single image frame, making predictions using the provided model and evaluating them against the ground truth.

    Args:
        model (object): The model object used to make predictions on the image frame.
        frame (np.ndarray): The image frame to be processed, represented as a NumPy array.
        ground_truth (list): A list of ground truth bounding boxes, where each bounding box is represented as a list of coordinates [x1, y1, x2, y2].
        iou_threshold (float): The Intersection over Union (IoU) threshold for determining whether a predicted bounding box matches the ground truth.

    Returns:
        np.ndarray: The annotated image frame with predictions overlaid.
        float: The accuracy of the model's predictions, calculated as the ratio of correct predictions to the total number of ground truth boxes.
    """
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

def getOutputPath(onnx_file: str,input:str, output:str)->str:
    """
    Function that takes input file path & output folder path and 
    returns the output file path.

    args:
        input (str) : input file path (/path/to/image.jpg)
        output (str) : output folder path (/path/to/output/)
    
    returns: 
        str : output file path (/path/to/output/image_output.jpg)
    """

    file_name, extension = os.path.splitext(os.path.basename(input))
    new_file_name = f"{file_name}_output{extension}"
    output_dir = output+'//'+onnx_file.split("\\")[-1]
    output_dir = os.path.normpath(output_dir)
    # import pdb;pdb.set_trace()
    ensure_directory_exists(output_dir)
    
    output_path =  os.path.normpath(os.path.join(output_dir, new_file_name))
    
    
    return output_path


def ensure_directory_exists(directory:str ):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): The path to the directory that needs to be checked.

    Functionality:
        - If the specified directory does not exist, it will be created.
        - If the directory already exists, no action is taken.
    """
    os.makedirs(directory, exist_ok=True)


def evaluate_models(onnx_files: list, img_list: list, gt_list: list, logging_folder: str, output_folder: str, iou_threshold: float, conf_threshold: float):
    """
    Evaluate a list of ONNX models on a set of images and ground truths, logging the performance metrics
    and identifying the model with the highest average accuracy.

    Args:
        onnx_files (list): List of paths to ONNX model files.
        img_list (list): List of paths to input images for evaluation.
        gt_list (list): List of paths to ground truth files corresponding to the images.
        logging_folder (str): Path to the folder where evaluation logs will be saved.
        output_folder (str): Path to the folder where infered and annotated images will be saved.
        iou_threshold (float): Intersection over Union (IoU) threshold for model evaluation.
        conf_threshold (float): Confidence threshold for model evaluation.

    Returns:
        str: Path to the ONNX model file with the highest average accuracy.
             Returns None if no models are evaluated.
    """
    best_accuracy = 0
    best_model = None
    
    # Initialize a default logger to avoid UnboundLocalError
    logger = logging.getLogger('default')

    for onnx_file in onnx_files:
        # Ensure logging directory exists
        if not os.path.exists(logging_folder):
            os.makedirs(logging_folder)
            
        # Create a logger for each model evaluation
        logger = logging.getLogger(onnx_file)
        logger.setLevel(logging.INFO)
        
        # Create a file handler for logging
        log_filename = f"{onnx_file.replace('.onnx', '')}_evaluation.log"
        log_filepath = os.path.join(logging_folder, log_filename.split('\\')[-1])
        log_filepath = os.path.normpath(log_filepath)
        # import pdb; pdb.set_trace()
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.INFO)
        
        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the file handler to the logger+
        logger.addHandler(file_handler)
        
        logger.info(f"Evaluating {onnx_file}...")
        
        print(f"Evaluating {onnx_file}...")
        model = OnnxRunner(onnx_file, nms_thresh=iou_threshold, conf_thresh=conf_threshold)
        total_accuracy = 0

        for idx, img_path in enumerate(img_list):
            ground_truth = load_ground_truth(gt_list[idx])
            image = cv2.imread(img_path)
            output_path = getOutputPath(onnx_file,img_path, output_folder)
            # import pdb;pdb.set_trace()
            annotated_frame, accuracy = ProcessFrame(model, image, ground_truth, iou_threshold)
            total_accuracy += accuracy
            cv2.imwrite(output_path, annotated_frame)
            
            logger.info(f"Processed image {img_path} with accuracy {accuracy:.4f}")

        avg_accuracy = total_accuracy / len(img_list)
        logger.info(f"Model: {onnx_file}, Average Accuracy: {avg_accuracy:.4f}")
        print(f"Model: {onnx_file}, Average Accuracy: {avg_accuracy:.4f}")

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_model = onnx_file
            
        logger.info(f"Completed evaluation for the {onnx_file}")
    
        # Remove the file handler after each iteration to avoid duplicate logs
        logger.removeHandler(file_handler)
        file_handler.close()
    
    if best_model:
        print(f"Best Model: {best_model} with Average Accuracy: {best_accuracy:.4f}")
        # return best_model
    else:
        print("No models were evaluated.")
        # return None

    # print(f"Best Model: {best_model} with Average Accuracy: {best_accuracy:.4f}")
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

    input_type = parse_input(args.input)  
    onnx_files = sorted(glob(os.path.join(args.onnx, "*.onnx")))

    if(input_type=="IMG"):
        
        img_list = [args.input]
        gt_list = [args.gt]

        # return

    if(input_type=="FOLDER"):

        img_list = sorted(glob(os.path.join(args.input,"*")))
        gt_list = sorted(glob(os.path.join(args.gt, "*")))

    best_model = evaluate_models(onnx_files, img_list, gt_list, args.logs, args.output, args.iou_threshold, args.conf_threshold)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Computer Vision Assignement')

    parser.add_argument('--input','-i', type=str, required=True, help="Data input path. Can be an image or a folder of images.")
    parser.add_argument('--onnx','-o', type=str, required=True, help="Onnx model path. Can be a single model or a directory of models")
    parser.add_argument('--output','-p', type=str, required=False, help="Output folder path. The infered and annotated data will be written in this folder. Default : results.", default="results")
    parser.add_argument('--iou_threshold','-u', type=float, required=False, help="IOU threshold used in NMS. Default : 0.4", default=0.4)
    parser.add_argument('--conf_threshold','-c', type=float, required=False, help="Confidence score threshold. Default : 0.5", default=0.3)
    parser.add_argument('--gt', '-g', type=str, required=True, help="Path to the ground truth text file/s.")
    parser.add_argument('--logs','-l', type=str, required=False, help="path to where logs are stored. Default : logs", default="logs")


    args = parser.parse_args()

    main(args)
