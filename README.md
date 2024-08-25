# computer-vision-task

## About The Project

This project involves utilizing the already provided ONNX files, evaluating them against the validation dataset i.e., by comparing the predicted bounding boxes to the ground truth data and identifying which model performs the best in terms of face detection. 

**Note: The best model is identified based on the _best average accuracy_.**

## Provided Resources

To implement the above project, the following resources are provided:
- A set of ONNX files found in [computer-vision-task/models/] folder
- The validation dataset can be found in [computer-vision-task/WIDERFACE_Validation/] folder which includes images along with the corresponding ground truth files
- A **main.py** script to load the ONNX files, image data, ground truth data, and evaluate and identify the best performing model.

## Dependencies

This section lists the required frameworks/libraries used to bootstrap the project.

- Anaconda/PyCharm (depending on the ease of usage)
- opencv-python-headless
- numpy
- onnxruntime
- fastapi
- uvicorn[standard]
- pydantic
- requests






