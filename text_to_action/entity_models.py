from pydantic import BaseModel, field_validator, Field
from typing import List, Union, Optional, Any, ClassVar
from dateutil import parser
import re
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import json

### Notes: 
# if there are multiple fields in the class, override init function to take an input str and process it to extract the required fields
# if there is only one field, use field_validator to extract the required fields

def extract_numeric(value):
    # find all digits and decimal points in the value
    numeric_part = ''.join(re.findall(r'\d+\.?\d*', value))
    return float(numeric_part)

def extract_string(value):
    str_part = ''.join(re.findall(r'[a-zA-Z]+', value))
    return str_part

def extract_unit(value):
    unit_part = ''.join(re.findall(r'[^\d.]+', value)).strip()
    return unit_part

def create_instance(class_name, params):
    cls = globals()[class_name]  # Access class by its name in global scope
    return cls(**params)        

# Parse the class name and parameters from the string representation
def parse_string_representation(s):
    parts = s.split('(', 1)
    class_name = parts[0]
    params_str = parts[1][:-1]  # Remove the closing parenthesis
    params = {}
    stack = []
    current_param = ""
    for char in params_str:
        if char == ',' and not stack:
            key, value = current_param.split('=')
            params[key.strip()] = eval(value.strip())
            current_param = ""
        else:
            if char in "[({":
                stack.append(char)
            elif char in "])}":
                stack.pop()
            current_param += char
    if current_param:  # Add the last parameter
        key, value = current_param.split('=')
        params[key.strip()] = eval(value.strip())
    return class_name, params

def llm_extract_parameters(text,param_type:type[BaseModel]):
    """
    Extract parameters from the text using an llm and initialize respictive class.
    
    Args:
        text : The text input.
        param_type : The type of parameter to extract.

    Returns: instances : The instances of the class of type param_type
    """

    field_names = list(param_type.model_fields.keys())

    fields_example = ", ".join([f"{field}=value{i+1}" for i, field in enumerate(field_names)])

    prompt = (
        f"Please analyze the following text to extract any information related to {param_type.description}. "
        f"Focus on extracting values for these specific fields: {', '.join(field_names)}. "
        f"Then, format the extracted information as Python code only for initializing instances of the specified class. "
        f"Here's the text:\n\n{text}\n\n"
        "Example of expected Python code output for initializing instances (your output should follow this structure): "
        f"[{param_type.__name__}({fields_example}), {param_type.__name__}({fields_example})]"
    )

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
        )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    result = chat_completion.choices[0].message.content

    # This pattern looks for a word (the class name), followed by an opening parenthesis,
    # then lazily matches any characters (the parameters) until it finds the closing parenthesis.
    pattern = r"\w+\(.*?\)"

    # Use re.findall to extract all class initializations
    class_initializations = re.findall(pattern, result)

    instances = []
    for class_initialization in class_initializations:
        class_name, params = parse_string_representation(class_initialization)
        if class_name == param_type.__name__:
            instances.append(create_instance(class_name, params))
    return instances

class CARDINAL(BaseModel):
    """Numerals that do not fall under another type"""
    value: Any
    description: ClassVar[str] = "Any cardinal numerals"

    @field_validator('value')
    @classmethod
    def get_numeric(cls,v) -> int:
        return extract_numeric(v)

class DATE(BaseModel):
    """Absolute or relative dates or periods"""
    date: str 
    description: ClassVar[str] = "Any absolute or relative dates or period in appropriate date format"

    @field_validator('date')
    @classmethod
    def check_date_format(cls, v):
        try:
            parsed_date = parser.parse(v, default=None)

            # Check specificity of the date and format accordingly
            if parsed_date.hour == 0 and parsed_date.minute == 0 and parsed_date.second == 0:
                # Date only
                return parsed_date.strftime('%Y%m%d')
            # Optionally, convert to a specific format, e.g., ISO 8601
            else:
                return parsed_date.strftime('%Y%m%dT%H')
        except ValueError:
            raise ValueError("Invalid date format")

class EVENT(BaseModel):
    """Named hurricanes, battles, wars, sports events, etc."""
    name: str
    description: ClassVar[str] = "Any named events"

class FAC(BaseModel):
    """Buildings, airports, highways, bridges, etc."""
    name: str
    description: ClassVar[str] = "Any named facilities like buildings, airports, highways, bridges, etc."

class GPE(BaseModel):
    """Countries, cities, states"""
    name: str
    description: ClassVar[str] = "Any named geographical locations like countries, cities, states"

class LANGUAGE(BaseModel):
    # LANGUAGE :  Any named language
    name: str
    description: ClassVar[str] = "Any named language"

class LAW(BaseModel):
    # LAW :  Named documents made into laws.
    name: str
    description: ClassVar[str] = "Any named documents made into laws"

class LOC(BaseModel):
    ## LOC :  Non-GPE locations, mountain ranges, bodies of water
    name: str
    description: ClassVar[str] = "Any named non-GPE locations like mountain ranges, bodies of water"

class MONEY(BaseModel):
    # MONEY :  Monetary values, including unit
    value: Union[float, int]
    currency: Optional[str] = 'USD'
    description: ClassVar[str] = "Any monetary values, including unit (default: USD if not found.)"

    def __init__(self, input_string: str):
        numeric_part = extract_numeric(input_string)
        unit_part = extract_unit(input_string)
        super().__init__(value=numeric_part, currency=unit_part)

class NORP(BaseModel):
    # NORP :  Nationalities or religious or political groups
    name: str
    description: ClassVar[str] = "Any named nationalities or religious or political groups"

class ORDINAL(BaseModel):
    # ORDINAL :  "first", "second", etc.
    value: str
    description: ClassVar[str] = "Any ordinal values like 'first', 'second', etc."

class ORG(BaseModel):
    # ORG :  Companies, agencies, institutions, etc.
    name: str
    description: ClassVar[str] = "Any named organizations like companies, agencies, institutions, etc."

class PERCENT(BaseModel):
    # PERCENT :  Percentage, including "%"
    value: Union[float, int] = Field(..., gt=0, lt=100)
    description: ClassVar[str] = "Any percentage values"

class PERSON(BaseModel):
    # PERSON :  People, including fictional
    name: str
    description: ClassVar[str] = "Any named persons, including fictional"

class PRODUCT(BaseModel):
    # PRODUCT :  Objects, vehicles, foods, etc. (not services)
    name: str
    description: ClassVar[str] = "Any named products like objects, vehicles, foods, etc."

class QUANTITY(BaseModel):
    # QUANTITY :  Measurements, as of weight or distance
    value: Union[float, int]
    unit: Optional[str]
    description: ClassVar[str] = "Any quantity measurements, as of weight or distance"

    def __init__(self, input_string: str=None, **kwargs):
        if input_string is None:
            super().__init__(**kwargs)
        else:
            numeric_part = extract_numeric(input_string)
            unit_part = extract_unit(input_string)
            super().__init__(value=numeric_part, unit=unit_part)

    
class TIME(BaseModel):
    # TIME :  Times smaller than a day
    time: str  # Assuming HH:MM:SS format
    description: ClassVar[str] = "Any time values in HH:MM:SS like format"

    @field_validator('time')
    @classmethod
    def check_time_format(cls, v):
        try:
            # Parse the string to datetime object
            parsed_time = parser.parse(v)
            # Extract the time part
            extracted_time = parsed_time.time()
            # Optionally, convert to a specific string format if needed
            return extracted_time.strftime('%H:%M:%S')
        except ValueError:
            raise ValueError("Invalid time format")

class WORK_OF_ART(BaseModel):
    # WORK_OF_ART :  Titles of books, songs, etc.
    name: str
    description: ClassVar[str] = "Any named works of art like titles of books, songs, etc."



