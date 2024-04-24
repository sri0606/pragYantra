
from .utils.image import Image

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

    def get_context(self):
        """
        Get the context of the visuals
        """
        #get the context of the visuals from image analysis
        return self.process_image()