import spacy
import inspect
from pydantic import BaseModel 
import text_to_action.actions as actions
from vector_utils.vector_emb import VectorStore, ModelSource
from text_to_action.entity_models import *

class ActionDispatcher:
    """
    Function Calling mechanism.
    """
    def __init__(self,vector_store_path):
        self.task_handler_context_store = VectorStore(embedding_model="all-MiniLM-L6-v2",
                                                      model_source = ModelSource.SBERT)
        self.task_handler_context_store.load(vector_store_path)
        # Load the spaCy model
        self.entity_recognizer = spacy.load("en_core_web_sm")

    @staticmethod
    def execute_action(function_name,extracted_parameters,text):
        """
        Execute the action with the extracted parameters.

        Args:
        function_name : The name of the function to execute.
        extracted_parameters [dict]: The extracted parameters from the text.
        text : The text input.
        """
        print(f"Executing function: {function_name.__name__} with the following parameters: {extracted_parameters}")
        # Get the signature of the function
        sig = inspect.signature(function_name)

        # Prepare a dictionary to hold the parameter instances/values
        arguments = {}

        # Track used values to ensure they are not reused
        used_values = {}

        for name, param in sig.parameters.items():
            param_type = str(param.annotation.__name__).upper()
            if param_type not in extracted_parameters and param_type not in ['STR']:
                extracted_parameters[param_type] = llm_extract_parameters(text, param.annotation)

        print("Extracted params:",extracted_parameters)
        for name, param in sig.parameters.items():
            param_type = str(param.annotation.__name__).upper()
            # Attempt to match the parameter with an extracted entity
            if param.annotation is str:
                arguments[name] = text 

            elif param_type in extracted_parameters:
                if param_type in used_values:
                    # If the parameter type has been used, use the next available entity
                    used_values[param_type] += 1
                else:
                    used_values[param_type] = 0
                
                if len(extracted_parameters[param_type]) <= used_values[param_type]:
                    # If all entities have been used, use the last one
                    arguments[name] = extracted_parameters[param_type][-1]
                else:
                    arguments[name] = extracted_parameters[param_type][used_values[param_type]]

            else:
                # If no matching entity was found, use a default value
                print("No matching entity found for", param_type)
                return

        # Unpack the args dictionary when calling the function
        result = function_name(**arguments)

        return result
    
    def dispatch(self, text,*args, **kwargs):
        """
        Dispatch the task to the appropriate functions.

        Args:
            text : The text input.

        Returns: results : The results of the function calls.
        """
        possible_actions = self.task_handler_context_store.query(text,k=5)
        extracted_params = self.extract_parameters(text)

        results = {}
        threshold = 0.45
        # threshold value for function selection
        if "threshold" in kwargs:
            threshold = kwargs["threshold"]     

        for action in possible_actions:
            if action[1] > threshold and action[0].id_name not in results:
                action_to_perform = getattr(actions, action[0].id_name)
                results[action[0].id_name] = self.execute_action(action_to_perform,extracted_params,text)
                break

        return results
        
    
    def extract_parameters(self,text):
        """
        Extracts the parameters from the text.
        """
        # Process the text
        doc = self.entity_recognizer(text)
        # Extract entities
        entities = {}
        for ent in doc.ents:
            entity_type = str(ent.label_).upper()
            value = ent.text

            if entity_type in globals():
                class_obj = globals()[entity_type]
                
                if issubclass(class_obj, BaseModel):
                    fields = list(class_obj.model_fields.keys())
                    if len(fields) == 1:
                        # If there's only one field, use it as the keyword argument
                        instance = class_obj(**{fields[0]: value})
                    else:
                        # directly pass the value
                        instance = class_obj(value)
                entities[entity_type] = [instance] if entity_type not in entities else entities[entity_type] + [instance]
        return entities


import time

if __name__ == "__main__":

    
    dispatcher = ActionDispatcher("misc_functions.h5")
    func_description = [
    {
        "name": "get_context_from_memory",
        "description": "Get the context from memory. Recall, retrieve the memory from past."
    },
    {
        "name": "get_weather",
        "description": "Get the weather, climate information."
    },
    {
        "name": "get_context_from_memory",
        "description": "Do you remember the last time we went to Hawaii?"
    },
    {
        "name": "get_weather",
        "description": "I wonder whats the weather like in New Zealand?"
    },
    {
        "name": "get_news",
        "description": "Get the latest news."
    },
    {
        "name": "get_stock",
        "description": "Get the stock information."
    },
    {
        "name": "get_news",
        "description": "Any latest headlines in AI?"
    },
    {
        "name": "get_stock",
        "description": "What company is most valued?"
    },
    {
        "name": "get_time",
        "description": "Check the current time."
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder for an event or task."
    },
    {
        "name": "get_time",
        "description": "What time is it in Tokyo right now?"
    },
    {
        "name": "set_reminder",
        "description": "Remind me to call mom at 3 PM."
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist."
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact."
    },
    {
        "name": "play_music",
        "description": "Can you play some jazz music?"
    },
    {
        "name": "send_message",
        "description": "Text John that I'll be late for the meeting."
    },
    {
        "name": "get_recipe",
        "description": "Find a recipe for a dish."
    },
    {
        "name": "book_flight",
        "description": "Book a flight to a destination."
    },
    {
        "name": "get_recipe",
        "description": "How do I make lasagna?"
    },
    {
        "name": "book_flight",
        "description": "I need to book a flight to Paris next month."
    },
    {
        "name": "translate_text",
        "description": "Translate text from one language to another."
    },
    {
        "name": "get_definition",
        "description": "Get the definition of a word."
    },
    {
        "name": "translate_text",
        "description": "How do you say 'thank you' in French?"
    },
    {
        "name": "get_definition",
        "description": "What's the meaning of 'ubiquitous'?"
    }
]

    # for i in func_description:
    #     store.add_vector(text=i["description"],id_name=i["name"])


     # Example queries
    query_texts = [
        "Recall the experience of hiking Mount Kilimanjaro.",
        "Is it hot in Arizona?",
        "Music and songs",
        "Is NVIDIA very rich?",
    ]
    dates = [None,None,1820,2021]
    for i,query in enumerate(query_texts):

        start_time = time.time()
        print(f"Query: {query}")
        results = dispatcher.dispatch(query,threshold=0.3)
        
        for result in results:
            print(result,":",results[result])

        print("\n")