import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import logging

# --- Pydantic Models for Response ---

class BoundingBox(BaseModel):
    """Pydantic model for a bounding box (x, y, width, height)."""
    x: int
    y: int
    w: int
    h: int

class VerificationResponse(BaseModel):
    """Pydantic model for the verification API response."""
    verification_result: str
    similarity_score: float
    bounding_boxes: dict[str, BoundingBox]
    message: str = "Verification successful"

# --- FastAPI Application ---

app = FastAPI(
    title="Face Verification API",
    description="Accepts two face images and returns if they are the same person.",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Function ---

async def read_image_from_uploadfile(file: UploadFile) -> np.ndarray:
    """Reads an uploaded file and decodes it into a CV2 image (numpy array)."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning(f"Failed to decode image: {file.filename}")
            raise HTTPException(status_code=400, detail=f"Could not decode image: {file.filename}")
        
        # Convert from BGR (OpenCV default) to RGB (DeepFace default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    except Exception as e:
        logger.error(f"Error reading file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {file.filename}")

# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Face Verification API is running. Go to /docs for API documentation."}


@app.post("/verify", response_model=VerificationResponse)
async def verify_faces(
    file1: UploadFile = File(..., description="First image for verification."),
    file2: UploadFile = File(..., description="Second image for verification.")
):
    """
    Verifies if two uploaded images contain the face of the same person.
    
    - **Detects** faces in both images using `retinaface`.
    - **Extracts** embeddings using `ArcFace`.
    - **Computes** similarity (cosine similarity).
    - **Returns** verification result, score, and bounding boxes.
    """
    
    # Read images in parallel (though IO is sequential here, processing can be)
    img1_rgb = await read_image_from_uploadfile(file1)
    img2_rgb = await read_image_from_uploadfile(file2)
    
    logger.info(f"Processing images: {file1.filename} and {file2.filename}")

    try:
        # Use DeepFace.verify
        # model_name = "ArcFace": State-of-the-art model for embeddings.
        # detector_backend = "retinaface": High-accuracy detector.
        # enforce_detection = True: Ensures we raise an error if a face isn't found.
        result = DeepFace.verify(
            img1_path=img1_rgb,
            img2_path=img2_rgb,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        
        # Process the result from DeepFace
        is_same_person = result["verified"]
        distance = result["distance"]
        
        # ArcFace uses cosine distance, so similarity = 1 - distance
        # Higher score = more similar
        similarity = 1.0 - distance
        
        # Extract bounding boxes
        bbox1_dict = result["facial_areas"]["img1"]
        bbox2_dict = result["facial_areas"]["img2"]

        logger.info(f"Verification complete for {file1.filename} and {file2.filename}. Same: {is_same_person}")

        # Create our Pydantic response object
        response = VerificationResponse(
            verification_result="same person" if is_same_person else "different person",
            similarity_score=similarity,
            bounding_boxes={
                "face1_bbox": BoundingBox(**bbox1_dict),
                "face2_bbox": BoundingBox(**bbox2_dict)
            }
        )
        return response

    except ValueError as e:
        # This exception is raised by deepface if no face is detected
        logger.warning(f"Face detection error: {e}")
        raise HTTPException(
            status_code=400, 
            detail="Error during face processing: Face could not be detected in one or both images."
        )
    except Exception as e:
        # Catch other potential errors
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # This block allows running the script directly with `python main.py`
    # However, for development, `uvicorn main:app --reload` is preferred.
    uvicorn.run(app, host="0.0.0.0", port=8000)