import requests
import base64

img_path = r"D:\Personal\Interviews\ML_SmartCowTakeHomeAssignment_2024\ML_SmartCowTakeHomeAssignment\Take_Home_Computer_Vision\computer_vision\computer-vision-task\WIDERFACE_Validation\images\4.jpg"

def encode_img(img_path:str):
    with open(img_path, "rb") as image_file:
    # Read the image data and encode it to base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

url = "http://localhost:8000/predict"
headers = {"Content-Type": "application/json"}
encoded_string = encode_img(img_path)
data = {
    "image": encoded_string
}
response = requests.post(url, json=data, headers=headers)

print(response.status_code)
print(response.json())