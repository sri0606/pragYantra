import sys
import logging
import warnings
from config import Config
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from text_to_action import ActionDispatcher
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(filename='chat.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def process_input(input_text,dispatcher:ActionDispatcher):
    try:
        if input_text.lower() == 'quit':
            print("\033[91mExiting chat...\033[0m") 
            sys.exit(0)  # Exit the application
        else:
            # Here, you can add your custom processing or function calls
            results = dispatcher.dispatch(text=input_text)
            for result in results:
                print(result,":",results[result])

    except Exception as e:
        # Log the error internally and display a generic message to the user
        logging.error("An error occurred: %s", str(e))
        print("Sorry, something went wrong. Please try again.")

def main():
    # Ignore specific FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("\033[94mInitializing chat...\033[0m") 

    context_filename = input("Enter the filepath of the action embeddings (like math.h5): ")
    actions_module = input("Enter the filepath of the corresponding actions module: ")
    is_verbose = input("Do you want to enable verbose output? (y/n): ")
    Config.set_verbose(is_verbose.lower() == 'y')
    dispatcher = ActionDispatcher(action_embedding_filename=context_filename,
                                  actions_filepath=actions_module)

    print("\033[92m\nWelcome to the terminal chat! Type 'quit' to exit.\033[0m")
    try:
        while True:
             # Set the color to cyan for the input prompt
            print("\033[96m>\033[0m ", end="")
            user_input = input().strip()

            process_input(user_input,dispatcher)
            print('\n')
    except KeyboardInterrupt:
        # Handle user interrupt (Ctrl+C) gracefully
        print("\nChat session ended.")
    except Exception as e:
        # Log unexpected errors and exit
        logging.error("An unexpected error occurred: %s", str(e))
        print("An unexpected error occurred. Exiting chat.")

if __name__ == "__main__":
    main()