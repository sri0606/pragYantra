from llama_cpp import Llama
# central processing unit for all the data coming from the sources


class Manas:
    """
    Manas class
    """
    def __init__(self):
        """
        Constructor
        """
        self.interpreter = Llama(model_path="./models/llama3_8B.gguf")
        pass

    def analyze(self, data):
        """
        Analyze the data
        """
        prompt = str(data)

        #The maximum number of tokens to generate. Shorter token lengths will provide faster performance.
        max_tokens = 200

        #The temperature of the sampling distribution. Lower temperatures will result in more deterministic outputs, while higher temperatures will result in more random outputs.
        temperature = 0.3

        #An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass. Lower values of top_p will result in more deterministic outputs, while higher values will result in more random outputs.
        top_p = 0.1

        #Whether to echo the prompt in the output.

        echo = True


        # Define the parameters
        model_output = self.interpreter(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )
        final_result = model_output["choices"][0]["text"].strip()