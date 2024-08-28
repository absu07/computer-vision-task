import requests
import base64
import argparse



# img_path = r".\WIDERFACE_Validation\images\4.jpg"     ##change the path according to needs

def encode_img(img_path:str):
    with open(img_path, "rb") as image_file:
    # Read the image data and encode it to base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def run_prediction(img_path):

    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}
    encoded_string = encode_img(img_path)
    data = {
        "image": encoded_string
    }
    response = requests.post(url, json=data, headers=headers)
    
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test the web  service with post request')
    parser.add_argument('--input','-i', type=str, required=True, help="Data input path.It should be a single image.") 
    args = parser.parse_args()
    img_path = args.input
    run_prediction(img_path)