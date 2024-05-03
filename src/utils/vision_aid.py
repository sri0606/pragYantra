import cv2
import torch
from torchvision import transforms
from PIL import Image as PILImage
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from datetime import datetime, timedelta
import json
import os
import time

# def get_classifier_labels():
#     """
#     Get the classifier labels
#     """
#     with open('./models/imagenet_class_index.json') as f:
#         data = json.load(f)
#         labels = {int(k): v[1] for k, v in data.items()}
#     return labels

# CLASSIFIER_LABELS = get_classifier_labels()

class ImageProcessor:
    """
    Image helper class for VisionAid and Vision classes
    """
    model = VisionEncoderDecoderModel.from_pretrained("models/gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("models/gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("models/gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    def __init__(self):
        """
        Constructor
        """
        # self._path = None
        # # Load the model weights
        # state_dict = torch.load('./models/vision_resnet18.pth')

        # # Load the model architecture, using ResNet18 as an example
        # self.classifiers = models.resnet18()
        # # Apply the weights to the model
        # self.classifiers.load_state_dict(state_dict)
        # # Ensure the model is in evaluation mode
        # self.classifiers.eval()
        pass

    def load_image(self, image_data):
        """
        Load an image from the given data
        """
        self._image = PILImage.fromarray(image_data)


    def classify_image(self):
        """
        Classify the image
        """

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(self._image)
        input_batch = input_tensor.unsqueeze(0)

        # Make sure the input tensor is on the same device as the model
        input_batch = input_batch.to(next(self.classifiers.parameters()).device)

        # Feed the image into the model
        with torch.no_grad():
            output = self.classifiers(input_batch)

        # The output is a tensor where each element is the predicted score for a class
        # To get the predicted class, we find the index of the maximum score
        _, predicted_class = torch.max(output, 1)

        # return CLASSIFIER_LABELS[predicted_class.item()]

    def detect_objects(self):
        """
        Detect objects in the image
        """
        pass

    def detect_faces(self):
        """
        Detect faces in the image
        """
        pass

    def detect_text(self):
        """
        Detect text in the image
        """
        pass

    def get_context(self,image_data_list):
        """
        Get the context of the visuals
        """
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        visual_context = self.__predict_image_caption(image_data_list,gen_kwargs)

        return visual_context
    
    def __predict_image_caption(self, image_data_list, gen_kwargs):
        """
        Predicts image captions for a list of image data.

        Args:
            image_data_list (list): A list of image data.
            gen_kwargs (dict): Keyword arguments for the caption generation.

        Returns:
            list: A list of predicted image captions.
        """
        images = []
        for image_data in image_data_list:
            i_image = PILImage.fromarray(image_data)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **gen_kwargs, pad_token_id=None)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

class VisionAid:
    """
    VisionAid class to handle all vision related tasks
    """
    def __init__(self,stop_event_wait_time=5, save_to_json_interval=3):
        """
        Initialize the VisionAid class.

        Parameters:
        stop_event_wait_time (int): The time to wait in seconds between checks of the stop event. Default is 5.
        save_to_json_interval (int): The interval in seconds at which to save the vision log to a JSON file. Default is 3.
        """
        self.stop_event_wait_time = stop_event_wait_time
        self.save_to_json_interval = save_to_json_interval
        self._camera = cv2.VideoCapture(0) # Initialize the camera
        self.image_processor = ImageProcessor()

    def __capture_image(self):
        """
        Capture an image from the camera
        """
        ret, frame = self._camera.read()  # Read a frame from the camera
        if not ret:
            raise IOError("Cannot read frame from camera")
        return frame

    def __preprocess_image(self):
        """
        Process the image
        """
        # Add your image processing 
        pass

    def get_image(self, save_path=None):
        """
        Get the image and optionally save it to a specified path
        """
        frame = self.__capture_image()
        self.__preprocess_image()
        if save_path is not None:
            cv2.imwrite(save_path, frame)  # Save the image to the specified path
        return frame

    def get_visual_context(self, stop_event):
        """
        Get the visual context.

        This method captures the visual context using an image capture mechanism and saves it to a JSON file.
        The visual context is captured at regular intervals and stored in the corresponding hour's entry in the JSON file.
        The JSON file is saved periodically based on the specified interval.

        Args:
            stop_event (threading.Event): An event object used to control the execution of the method.

        Returns:
            None
        """
        transcript_dir_path = "memory_stream/vision_logs/"

        # Initialize a timer
        next_save_time = datetime.now() + timedelta(seconds=self.save_to_json_interval)
        vision_log = {}

        while not stop_event.is_set():
            now = datetime.now()
            current_hour = now.strftime("%H")
        
            print("Getting image...")
            image_start_time = time.time()
            image = self.get_image()
            image_end_time = time.time()
            print(f"Got image in {image_end_time - image_start_time} seconds")

            print("Getting context from image...")
            context_start_time = time.time()
            context = self.__get_context_from_image(image)
            context_end_time = time.time()
            print(f"Got context in {context_end_time - context_start_time} seconds")

            vision_log[current_hour] = [{now.strftime("%H%M%S"): context}]

            if current_hour not in vision_log:
                    vision_log[current_hour] = []

            if now >= next_save_time:
                date_string = now.strftime("%Y%m%d")
                filename = f"{date_string}_vision.json"
                filepath = os.path.join(transcript_dir_path, filename)

                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        if f.read().strip():
                            f.seek(0)  # reset file pointer to beginning
                            existing_data = json.load(f)
                        else:
                            existing_data = {}
                else:
                    existing_data = {}

                if current_hour not in existing_data:
                    existing_data[current_hour] = []

                existing_data[current_hour].extend(vision_log[current_hour])

                with open(filepath, 'w') as f:
                    json.dump(existing_data, f)

                vision_log[current_hour].clear()

                next_save_time = now + timedelta(seconds=self.save_to_json_interval)
            # Wait for 5 seconds or until the stop event is set
            stop_event.wait(self.stop_event_wait_time)    

    def __get_context_from_image(self, image_paths):
        """
        Get the context of the visuals
        """
        context = []
        for image_path in image_paths:
            # self.image.load(image_path)
            context.append(self.image_processor.get_context(image_path))
        return context
    
    def __del__(self):
        """
        Destructor
        """
        self._camera.release()  # Release the camera when the object is destroyed
    
    def process_image(self):
        """
        Process the image
        """
        results = {
            'object_detection': self.image.detect_objects(),
            'face_detection': self.image.detect_faces(),
            'text_detection': self.image.detect_text(),
        }
        return results