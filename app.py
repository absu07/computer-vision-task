import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from typing import List, Tuple
from utils import OnnxRunner

app = FastAPI()

# Define the data model for the request body
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

# Helper functions from your existing code
def preProcessFrame(frame: np.ndarray) -> np.ndarray:
    # Resize the frame to 640x640
    frame_resized = cv2.resize(frame, (640, 640))
    
    # Convert frame color channel from BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Standardize the pixel values
    frame_standardized = (frame_rgb - 127.5) / 127.5
    
    # Change data format to BCHW (Batch, Channels, Height, Width)
    frame_bchw = np.transpose(frame_standardized, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    
    return frame_bchw

@app.post("/predict")
async def predict_faces(request: ImageRequest):
    try:
        # Decode the base64 image
        img_data = base64.b64decode(request.image)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Process the image
        processed_frame = preProcessFrame(image)
        onnx_model = OnnxRunner(r"D:\Personal\Interviews\ML_SmartCowTakeHomeAssignment_2024\ML_SmartCowTakeHomeAssignment\Take_Home_Computer_Vision\computer_vision\computer-vision-task\models\24.onnx")
        import pdb;pdb.set_trace()
        metadata = onnx_model.run(processed_frame)

        # Process the metadata to extract bounding boxes
        height, width, _ = image.shape
        bounding_boxes = []
        for box in metadata:
            # import pdb;pdb.set_trace()
            x1, y1, x2, y2, score = box
            # Convert normalized coordinates to pixel values
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            bounding_boxes.append([x1, y1, x2, y2, score])

        return {"bounding_boxes": bounding_boxes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using Uvicorn if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
