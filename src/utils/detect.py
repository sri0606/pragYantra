import cv2
import numpy as np
import tensorflow as tf
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import easyocr

def detect_faces(image_path):
    # Load the pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces

def detect_objects(image_path):
    # Load the pre-trained object detection model from TensorFlow
    detection_model = tf.saved_model.load('path/to/saved_model')
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Preprocess the image
    resized_image = cv2.resize(image, (300, 300))
    input_tensor = tf.convert_to_tensor(resized_image)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Perform object detection
    detections = detection_model(input_tensor)
    
    return detections

def detect_text(image_path):
    
    reader = easyocr.Reader(['en'],gpu=False) # this needs to run only once to load the model into memory
    result = reader.readtext(image_path)
    
    return result

def extract_text(image_path):
    reader = easyocr.Reader(['en'],gpu=False) # this needs to run only once to load the model into memory
    result = reader.readtext(image_path, detail=0)
    
    return result

def extract_text_handwritten(image_path):
    
    # Load the processor and model from a specific folder
    processor = TrOCRProcessor.from_pretrained('../models/trOCR')
    model = VisionEncoderDecoderModel.from_pretrained('../models/trOCR')

    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text



