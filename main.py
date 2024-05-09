import argparse
import os
import time
from dotenv import load_dotenv
load_dotenv()
import logging
from transformers import logging as hf_logging
logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()

def main():
    parser = argparse.ArgumentParser(description='Virtual Assistant')
    parser.add_argument('--interpreter_model', type=str, default='llama3-70B-8192', help='The name of the groq model to use.')
    parser.add_argument('--offline_mode', action='store_true', help='Whether to run in offline mode. Default is True.')
    parser.add_argument('--speaker_model', type=str, default='pyttsx3', help='The speaker model to use.')
    
    args = parser.parse_args()
    
    if not args.offline_mode:
        groq_api_key = os.getenv('GROQ_API_KEY')
        if groq_api_key is None:
            print("Please set the GROQ_API_KEY environment variable to use online mode.")
            return
    else:
        model_path = os.path.join("models", args.interpreter_model+".gguf")
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Please check the model name.")
            return
        
    if args.speaker_model not in ["pyttsx3", "11labs", "facebook-mms"]:
        print("Invalid speaker model. Please choose from 'pyttsx3', '11labs', or 'facebook-mms'.")
        return
    if args.speaker_model == "11labs":
        speech_key = os.getenv('SPEECH_KEY')
        if speech_key is None:
            print("Please set the SPEECH_KEY environment variable in a '.env' file to use Eleven Labs for speech model.")
            return
    
    print("\nHello! \nSetting up the pragYantra virtual assistant for you...\n")
    from core.manas import Manas
    robo = Manas(interpreter_model=args.interpreter_model, offline_mode=args.offline_mode, speaker_model=args.speaker_model)

    robo.setup()
    robo.start()

    # keep the program running
    try:
        print("\nSetup complete. You can interact with it now. There might be a slight delay before it can respond initially. \nPress Ctrl+C to terminate.\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\Terminating...\n")
        # terminate when user presses Ctrl+C,
        robo.terminate()
        print("\nSee ya..Bye!")
if __name__ == "__main__":
    main()