
from .utils.image import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("models/gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("models/gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("models/gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class VisionAid:
    """
    VisionAid class to handle all vision related tasks
    """
    def __init__(self):
        """
        Constructor
        """
        # self._camera = Camera()
        self._image = None

    def __capture_image(self):
        """
        Capture an image from the camera
        """
        self._image = self._camera.capture_image()

    def __preprocess_image(self):
        """
        Process the image
        """
        pass

    def get_image(self):
        """
        Get the image
        """
        self.__capture_image()
        self.__preprocess_image()
        return self._image
    
class Vision:
    """
    Vision class to handle all vision related tasks
    """
    def __init__(self):
        """
        Constructor
        """
        self._eyes = VisionAid()
        self.image = Image()

    def capture_image(self):
        """
        Capture an image from the camera
        """
        self.image.load(self._eyes.get_image())

    def process_image(self):
        """
        Process the image
        """
        results = {
            'image_classification': self.image.classify_image(),
            'object_detection': self.image.detect_objects(),
            'face_detection': self.image.detect_faces(),
            'text_detection': self.image.detect_text(),
        }

        return results

    def get_context(self,image_paths):
        """
        Get the context of the visuals
        """
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        visual_context = self.__predict_image_caption(image_paths,gen_kwargs)

        return visual_context
    
    def __predict_image_caption(self,image_paths,gen_kwargs):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs,pad_token_id=None)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds