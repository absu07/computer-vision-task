# computer-vision-task

## About The Project

This project involves utilizing the already provided ONNX files, evaluating them against the validation dataset i.e., by comparing the predicted bounding boxes to the ground truth data and identifying which model performs the best in terms of face detection. 

**Note: The best model is identified based on the _best average accuracy_.**


## Provided Resources

To implement the above project, the following resources are provided:
- A set of ONNX files found in [computer-vision-task/models/] folder
- The validation dataset can be found in [computer-vision-task/WIDERFACE_Validation/] folder which includes images along with the corresponding ground truth files
- A **main.py** script to load the ONNX files, image data, ground truth data, and evaluate and identify the best-performing model.


## Dependencies

This section lists the required frameworks/libraries used to bootstrap the project.

- Anaconda/PyCharm (depending on the ease of usage)
- Docker Desktop (If you are using Windows)
- opencv-python-headless
- numpy
- onnxruntime
- fastapi
- uvicorn[standard]
- pydantic
- requests

## Set-Up and Installation

This section describes the steps to install the prerequisites and set-up the project required for running the application.

### Prerequisites

Before you begin, make sure you have the following software installed on your system:

- **Python 3.8 or higher**: Required for running the project's scripts. [Download Python](https://www.python.org/downloads/)
- **Docker 20.x**: Required for containerization if you prefer running the project in a Docker environment. [Install Docker](https://docs.docker.com/get-docker/)
- **Anaconda** (optional): Recommended for managing Python environments. [Download Anaconda](https://www.anaconda.com/products/distribution)

Ensure that these tools are properly installed and configured before proceeding to the installation steps.

### Setting up the project and installing the dependencies

Follow these steps to set up the project on your local machine:

1. Clone the Repository:
First, clone the repository to your local machine using Git.

   git clone https://github.com/absu07/computer-vision-task.git

2. Navigate to the Project Directory:
Move into the project directory.

   `cd computer-vision-task`

3. Create a Virtual Environment:
Create a new virtual environment using Anaconda with Python 3.12.4.

   `conda create --name myenv python=3.12.4`
   
5. Activate the Virtual Environment:
Activate the newly created environment.

   `conda activate myenv`

6. Install Required Dependencies:
Install all the necessary Python packages listed in requirements.txt.

   `pip install -r requirements.txt`

## Usage

This section describes the steps to execute the application (either as a standalone or inside a container).

1. Running the application as a standalone.
To run the script, use the following command:

`python main.py -i "Data input path" -o "ONNX model path" -g "Path to the ground truth text files"`

2. Running the application inside the docker container.
To run the script inside the container, use the following commands:

   1. Build the Docker Image:
  
   `docker build -t "image-name" .`

   2. Running the Docker Container:
   Once the Docker image is built, you can run the container with the following command:

   `docker run --rm -it "image-name"`

   This command starts the application which evaluates the ONNX files against the validation data provided and identifies the best-performing model.









