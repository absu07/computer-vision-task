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
        print(f"x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}, score {score}")

        # Convert normalized coordinates to pixel values
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        print("Drawing bounding box...")
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print("Bounding box drawn successfully.")

        # Put score text
        label = f"{score:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Predictions', annotated_frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return annotated_frame


def ProcessFrame(model:object,frame:np.ndarray):
    """
    Main function that will take a raw OpenCV frame,
    will preprocess it, will infer it using the onnx model
    and will postprocess the result in order to draw the 
    bounding boxes on the frame and save it.

    args : 
        model (OnnxRunner) : OnnxRunner that wraps an ORT object, able to be infered
        frame (np.ndarray) : raw frame HWC

    returns : 
        np.ndarray : same frame as input but with bounding boxes drawn on it
    """

    processed_frame = preProcessFrame(frame)
    metadata = model.run(processed_frame)
    print(f"Metadata: {metadata}")
    annotated_frame = postProcessFrame(metadata, frame)

    return annotated_frame

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

    file_name, extension = input.split("/")[-1].split(".")
    return os.path.join(output,file_name+"_output."+extension)


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

    model = OnnxRunner(args.onnx, nms_thresh=args.iou_threshold, conf_thresh=args.conf_threshold)

    input_type = parse_input(args.input)

    if(input_type=="IMG"):

        print(f"[INFO] Processing 1 image : {args.input}")

        TIME_FLAG = time.time()

        image = cv2.imread(args.input)
        output_path = getOutputPath(args.input, args.output)
        annotated_frame = ProcessFrame(model, image)

    
        print(f"[INFO] Processed image in {time.time()-TIME_FLAG}s")

        cv2.imwrite(output_path, annotated_frame)

        return

    if(input_type=="FOLDER"):

        img_list = glob(os.path.join(args.input,"*"))

        print(f"[INFO] Processing a folder containing {len(img_list)} images.")

        TIME_FLAG = time.time()

        for img_path in img_list:
            
            image = cv2.imread(img_path)
            output_path = getOutputPath(img_path, args.output)
            annotated_frame = ProcessFrame(model, image)
            cv2.imwrite(output_path, annotated_frame)

        print(f"[INFO] Processed {len(img_list)} images in {time.time()-TIME_FLAG}s. Avg : {(time.time()-TIME_FLAG)/len(img_list)}s")

        return
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Computer Vision Assignement')

    parser.add_argument('--input','-i', type=str, required=True, help="Data input path. Can be an image or a folder of images.")
    parser.add_argument('--onnx','-o', type=str, required=True, help="Onnx model path.")
    parser.add_argument('--output','-p', type=str, required=False, help="Output folder path. The infered and annotated data will be written in this folder. Default : results.", default="results")
    parser.add_argument('--iou_threshold','-u', type=float, required=False, help="IOU threshold used in NMS. Default : 0.4", default=0.25)
    parser.add_argument('--conf_threshold','-c', type=float, required=False, help="Confidence score threshold. Default : 0.5", default=0.25)


    args = parser.parse_args()

    main(args)
