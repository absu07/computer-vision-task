# [<img src="https://uploads-ssl.webflow.com/62fcb047012c931f05d357dc/64ac1e629d8e46c513ea3d5d_SmartCow%20Logo.svg">](https://www.smartcow.ai/) AI Take Home Assignment

Choose **ONE** of the two tasks below, 

1. [Computer Vision](#computer-vision-task)

2. [Machine Learning](#machine-learning-task)

## [Computer Vision Task](./Take_Home_Computer_Vision)

It is the year 2077, cows have evolved and have learnt how to disguise themselves as humans. Engineers at SmartCow have developed a reliable method to identify these human-cow imposters based on **Face Detection**. 

However, a *cow spy*, renamed all the [ONNX](https://onnx.ai/) checkpoints of the trained model and because of this we lost track of which ONNX file corresponds to the best performing model. 

There is no time to re-train the model, your task is to use the validation data provided and identify which out of all the provided ONNX files performs the best.

### Provided Resources

To complete the task you are provided the with following resources:
- A set of ONNX files found in [Take_Home_Computer_Vision/models/](./Take_Home_Computer_Vision/models/)
- A subset of images from the validation set of the [WIDERFACE Dataset](http://shuoyang1213.me/WIDERFACE/) found in [Take_Home_Computer_Vision/WIDERFACE_Validation](./Take_Home_Computer_Vision/WIDERFACE_Validation)
  - Each image in the validation set has a corresponding label file containing the ground truth bounding boxes.
- A [Template Script](./Take_Home_Computer_Vision/main.py) to assist you in loading the ONNX files and post process the predictions.

### Instructions

- Load an ONNX file
- Preprocess the images to conform with the the below requirements:
  - The image color channel order must be in **`RGB`**
  - The image size must be *`640x640`*
  - The pixel values must be standardised using **`mean=127.5`** and **`std=127.5`**
  - The image must be in **`CHW`** Format
- Infer on the validation images
- Evaluate the ONNX file by comparing the predicted bounding boxes to the provided ground truth.

### Deliverables

Send a zipped repository with a solution (make sure it contains .git directory to consult history of commits)

Write your code in [main.py](./Take_Home_Computer_Vision/main.py)

You are to provide:
- The complete `main.py` script and any other scripts you write, with your amended code.
- Documentation **justyfying** your results when comparing the ONNX files.
- Dockerise your solution:
  - Create a standalone [Docker](https://www.docker.com/) container that will install and setup all the requirements used by your application and is able to execute your scripts. 
  - Using a web application framework such as [Fast API](https://fastapi.tiangolo.com/) or [Flask](https://flask.palletsprojects.com/) convert your application into a web service that is able to perform predictions through a web request:
    - The service must receive a [base64](https://en.wikipedia.org/wiki/Base64) image string and return **a list of bounding boxes** of the faces in the image.

Please note that your code will be executed on a **Linux-based** machine.

## [Machine Learning Task](./Take_Home_Machine_Learning/)

Farmers at SmartCow have begun to recount an increasingly concerning amount of UFO sightings. There seems to be an **underlying pattern**, so they have decided to contact you, a prospective SmartCow Machine Learning Engineer, to help them out.

Your task is to use the provided datasets and create **TWO** distinct Machine Learning models to **predict** if a UFO will appear.

Deep Learning is not required for this task, however, it is encouraged.

### Data Specifications

There are three datasets which you are to use for this task:

- [main.csv](Take_Home_Machine_Learning/csv/main.csv) - holds information on both animals and objects on a per-frame basis, including the total count of animals and objects per frame.
- [animals.csv](Take_Home_Machine_Learning/csv/animals.csv) - holds all information related to animals detected within a frame, including the number of animals, their age in days, and their bounding boxes within the frame.
- [objects.csv](Take_Home_Machine_Learning/csv/objects.csv) - holds all information related to stationary objects, odd extraterrestrial life (aliens), and the ever-elusive UFO sighting within a single frame.

### Instructions

- Install requirements

    ```pip3 install -r requirements.txt```

- Run the application

    ```python3 main.py```

### Deliverables

Send a zipped repository with a solution (make sure it contains .git directory to consult history of commits)

Write your code either in [main.py](./Take_Home_Machine_Learning/main.py) where indicated or in a separate class file and call your class where indicated in `main.py`. Please include appropriate comments and explain your thought process where possible. 

You are to provide:
- The scripts with your amended code.
- Two `.csv` files: One containing your processed and filtered features, one containing your predictions.
- Files (in `.pkl` or otherwise) containing the `models` which you used for this task.
- A short documentation of your approach where you **demonstrate** some Exploratory Data Analysis, **justify** your choice in Machine Learning algorithms and the **Evaluation Metrics** selected for your solution.  
- Dockerise your solution:
    - Create a standalone [Docker](https://www.docker.com/) container that will install and setup all the requirements used by your application and is able to execute your scripts. 
    - Using a web application framework such as [Fast API](https://fastapi.tiangolo.com/) or [Flask](https://flask.palletsprojects.com/) convert your application into a web service that is able to perform predictions through a web request:
      - The service must receive the model input features as a request and return a **boolean** whether or not a UFO will appear.

Please note that your code will be executed on a **Linux-based** machine.
