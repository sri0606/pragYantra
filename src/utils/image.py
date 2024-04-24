import torch
from torchvision import models, transforms
from PIL import Image
import json

def get_classifier_labels():
    """
    Get the classifier labels
    """
    with open('./models/imagenet_class_index.json') as f:
        data = json.load(f)
        labels = {int(k): v[1] for k, v in data.items()}
    return labels

CLASSIFIER_LABELS = get_classifier_labels()

class Image:
    """
    Image helper class for VisionAid and Vision classes
    """
    def __init__(self):
        """
        Constructor
        """
        self._path = None
        # Load the model weights
        state_dict = torch.load('./models/vision_resnet18.pth')

        # Load the model architecture, using ResNet18 as an example
        self.classifiers = models.resnet18()
        # Apply the weights to the model
        self.classifiers.load_state_dict(state_dict)
        # Ensure the model is in evaluation mode
        self.classifiers.eval()

    def load_image(self, path):
        """
        Load an image from the given path
        """
        self._path = path

    def classify_image(self):
        """
        Classify the image
        """
        # Load and preprocess the image
        image = Image.open(self._path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # Make sure the input tensor is on the same device as the model
        input_batch = input_batch.to(next(self.classifiers.parameters()).device)

        # Feed the image into the model
        with torch.no_grad():
            output = self.classifiers(input_batch)

        # The output is a tensor where each element is the predicted score for a class
        # To get the predicted class, we find the index of the maximum score
        _, predicted_class = torch.max(output, 1)

        return CLASSIFIER_LABELS[predicted_class.item()]

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
