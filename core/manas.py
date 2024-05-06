from llama_cpp import Llama
from hear import LiveTranscriber
from vision import LiveVision
from speech import LiveSpeech
from memory import LiveMemory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import time 
import os

class Interpreter:
    """
    Interpreter class
    """
    def __init__(self):
        """
        Constructor
        """
        pass
    
    def get_response(self, prompt):
        """
        Get response from the interpreter.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            str: The generated response from the model.
        """
        return NotImplementedError("Subclass must implement get_response method")
    
class Llama(Interpreter):
    """
    Llama class is an offline interpreter. It can be slow depending on the model size and device specs.
    It inherits from the Interpreter base class and provides an implementation for the get_response method.
    """
    def __init__(self, model_path="./models/llama3_8B.gguf"):
        """
        Constructor for the Llama class.

        Args:
            model_path (str): Path to the model file. Defaults to "./models/llama3_8B.gguf".
        """
        self._llama = Llama(model_path=model_path)
    
    def get_response(self, prompt):
        """
        Get response from the llama model.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            str: The generated response from the model.
        """
        #max_tokens: The maximum number of tokens to generate. Shorter token lengths will provide faster performance.

        #temperature: The temperature of the sampling distribution. Lower temperatures will result in more deterministic outputs, 
        #while higher temperatures will result in more random outputs.

        #top_p: An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass. 
        #       Lower values of top_p will result in more deterministic outputs, while higher values will result in more random outputs.

        #echo: Whether to echo the prompt in the output.

        # Define the parameters
        model_output = self.interpreter(
            prompt,
            max_tokens=200,
            temperature=0.3,
            top_p=0.1,
            echo=False,
        )
        final_result = model_output["choices"][0]["text"].strip()

        return final_result
    
class Groq(Interpreter):
    """
    Groq class. Online and super fast.

    This class represents a Groq interpreter that can be used to interact with the Groq API.
    It provides methods to initialize the interpreter and get responses from the Groq API.

    Attributes:
        model_name (str): The name of the Groq model to use.
        groq_api_key (str): The API key for accessing the Groq API.
        temperature (float): The temperature parameter for generating responses. Default is 0.

    Methods:
        __init__(model_name, groq_api_key, temperature=0): Initializes the Groq interpreter.
        get_response(prompt): Sends a prompt to the Groq API and returns the response.

    Example usage:
        groq = Groq(model_name='my_model', groq_api_key='my_api_key', temperature=0.5)
        response = groq.get_response('Hello, how are you?')
    """
    def __init__(self, model_name, groq_api_key=None, temperature=0):
        """
        Constructor

        Args:
            model_name (str): The name of the Groq model to use.
            groq_api_key (str): The API key for accessing the Groq API. If not provided, it will be read from the environment variable GROQ_API_KEY.
            temperature (float, optional): The temperature parameter for generating responses. Default is 0.
        """
        if groq_api_key is None:
            groq_api_key = os.getenv("GROQ_API_KEY")

        self._groq = ChatGroq(temperature=temperature, model_name=model_name, groq_api_key=groq_api_key)
    
    def get_response(self, prompt):
        """
        Get response from the groq

        Args:
            prompt (str): The prompt to send to the Groq API.

        Returns:
            str: The response from the Groq API.
        """
        prompt = ChatPromptTemplate.from_messages([("system", prompt)])

        chain = prompt | self._groq
        model_output2=  chain.invoke({})
        final_result= model_output2.content
        return final_result
    
# central processing unit for all the data coming from the sources
class Manas:
    """
    Manas class represents a virtual brain that can process multiple inputs from different sources like eyes and ears.
    It can see, hear, speak, and understand the world around it.
    """

    def __init__(self):
        """
        Constructor
        """
        self.interpreter = Groq(model_name="llama3-70b-8192")
        self.ears = LiveTranscriber()
        self.eyes = LiveVision()
        self.speak = LiveSpeech(speaker_model="pyttsx3")
        self.memory = LiveMemory(latest_time_threshold=7)
        self.alive = False
        return

    def start(self):
        """
        Start the Manas
        """
        self.ears.start()
        self.eyes.start()
        self.speak.start()
        self.alive = True
        return

    def terminate(self):
        """
        End the Manas
        """
        self.ears.terminate()
        self.eyes.terminate()
        self.speak.terminate()
        self.alive = False
        return

    def analyze(self, data,previous_repsonse):
        """
        Analyze the data

        Args:
            data (dict): A dictionary containing the vision and audio data.
            previous_repsonse (str): The previous response
        Returns:
            str: The conversational response generated by the Manas.
        """

        prompt = f"""
        You are like a virtual brain of a human processing multiple inputs from different sources like eyes, ears. You can see, hear, speak and understand the world around you. 

        Here is what you seen and heard recently about 2-3 seconds ago:

        What you saw: {data['recent']['vision']}

        What you heard: {data['recent']['audio']}

        Your response was: "{previous_repsonse}"

        
        Here is what you see and hear currently:
        
        What you see: {data['latest']['vision']}

        What you hear: {data['latest']['audio']}

        Using that current information, generate a conversational response as if you are a human, talking.
        """

        response = self.interpreter.get_response(prompt)

        return response

    def live(self):
        """
        Bring it to life
        """
        response=""
        self.speak.set_ears_pause_event(self.ears.get_pause_event())
        
        while self.alive:
            # Check if the speech is running, if not, analyze the data and speak
            if not self.speak.is_speaking():
                # Check if someone is talking, if so, pause the speech
                if self.ears.is_voice_detected():
                    self.speak.pause()
                    time.sleep(1)
                    continue

                data = self.memory.get_latest_memory()
                if data["latest"]["audio"]: #and data["latest"]["vision"] 
                    response = self.analyze(data,previous_repsonse=response)
                    # self.ears.pause()
                    self.speak.speak(response)
                    # self.ears.resume()
                    time.sleep(1)
                    
        return
    
if __name__ == '__main__':
    manas = Manas()
    # manas.start()

    data = {"vision":"A man laying on a bed with his pillow",
            "audio" :"Do you know that I bought this pillow for $5! Haha!"}
    response = manas.analyze(data)

    print(response)