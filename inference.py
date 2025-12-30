import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import time
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(r"c:\Users\nchar\OneDrive\Desktop\eye_disease_detection")
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pth"  # Or final_model.pth
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['healthy', 'unhealthy']

# --- Model Loading ---
# Removed cache to ensure latest model is always loaded during dev
def load_model():
    print("Loading model from disk...")
    # Initialize EfficientNet B0 (No weights needed, we'll load ours)
    model = models.efficientnet_b0(weights=None) 
    
    # Match the head structure
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    # Load weights
    if MODEL_PATH.exists():
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            return None
    else:
        st.error(f"Model file not found at {MODEL_PATH}")
        return None

    model = model.to(DEVICE)
    model.eval()
    return model

# --- Preprocessing ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(model, image):
    image_tensor = data_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    class_name = CLASSES[preds[0]]
    probability = probs[0][preds[0]].item()
    return class_name, probability

# --- Main App ---
def main():
    st.title("Eye Disease Detection App")
    st.sidebar.title("Navigation")
    
    mode = st.sidebar.radio("Choose Mode", ["Upload Image", "Real-time Camera"])
    
    model = load_model()
    if model is None:
        return

    # Load Haar Cascade
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if mode == "Upload Image":
        st.header("Upload an Eye Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("Analyze"):
                with st.spinner('Analyzing...'):
                    # Convert to numpy for OpenCV processing (RGB)
                    img_np = np.array(image)
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    
                    # Detect eyes
                    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(eyes) > 0:
                        st.write(f"Detected {len(eyes)} eye(s).")
                        annotated_img = img_np.copy()
                        
                        for i, (x, y, w, h) in enumerate(eyes):
                            # Crop eye from original image (PIL)
                            eye_roi = image.crop((x, y, x+w, y+h))
                            
                            # Predict
                            class_name, prob = predict_image(model, eye_roi)
                            
                            # Draw box on annotated image (RGB)
                            if class_name == 'healthy':
                                color = (0, 255, 0) # Green
                            else:
                                color = (255, 0, 0) # Red
                            
                            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
                            cv2.putText(annotated_img, f"Eye {i+1}", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            
                            # Display Result Text
                            res_text = f"Eye {i+1}: {class_name.upper()} ({prob*100:.2f}%)"
                            if class_name == 'healthy':
                                st.success(res_text)
                            else:
                                st.error(res_text)
                        
                        # Show annotated image
                        st.image(annotated_img, caption='Analyzed Image with Detections', use_column_width=True)
                        
                    else:
                        st.warning("No eyes detected. Analyzing entire image.")
                        class_name, prob = predict_image(model, image)
                        if class_name == 'healthy':
                            st.success(f"Prediction: {class_name.upper()} ({prob*100:.2f}%)")
                        else:
                            st.error(f"Prediction: {class_name.upper()} ({prob*100:.2f}%)")

    elif mode == "Real-time Camera":
        st.header("Real-time Eye Verification")
        st.write("Press 'Start' to open webcam. Press 'Stop' in the sidebar or below to end.")
        
        run_camera = st.checkbox('Start Camera')
        FRAME_WINDOW = st.image([])
        
        camera = cv2.VideoCapture(0)
        
        while run_camera:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break
            
            # Convert for OpenCV processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in eyes:
                # Draw box
                color = (0, 255, 0) # Green default
                
                # Crop and Predict
                eye_roi = frame[y:y+h, x:x+w]
                try:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    eye_pil = Image.fromarray(cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB))
                    class_name, prob = predict_image(model, eye_pil)
                    
                    # Update Color and Text based on prediction
                    label = f"{class_name} ({prob:.2f})"
                    if class_name == 'unhealthy':
                        color = (0, 0, 255) # Red
                    else:
                        color = (0, 255, 0) # Green
                        
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                except Exception as e:
                    # ROI might be too small
                    pass

            # Convert to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
            # Allow Streamlit to handle UI events? Not easy in loop.
            # Usually check st state.
            if not run_camera:
                break
        
        camera.release()

if __name__ == '__main__':
    main()
