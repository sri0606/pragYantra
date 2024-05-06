# PragYantra

PragYantra is a versatile software project that aims to simulate a humanoid robot with vision, hearing, speech, and memory functionalities. This project aims to create a flexible platform for experimenting with artificial intelligence and human-machine interaction.

![PragYantra's Architecture](arch.jpg)

## Features

- **Vision**: PragYantra simulates vision capabilities, allowing it to process visual data and make decisions based on it.
- **Hearing**: With simulated auditory sensors, PragYantra can perceive sounds and respond accordingly.
- **Speech**: PragYantra is capable of generating simulated speech output, enabling it to communicate with users in natural language.
- **Memory**: The software includes memory capabilities, enabling PragYantra to store and recall information from previous interactions.

## Technical details

One of the key design principles behind PragYantra was prioritizing offline capabilities while also integrating online functionalities. This decision was made to ensure privacy and accessibility without compromising on performance.

To achieve this, all components of the project were designed to have offline capabilities, with online functionalities available as optional features. While using offline mode may require a stronger device for faster inference, the project is fully functional and performs admirably under these conditions.

The backbone of PragYantra consists of various open-source models for tasks such as text-to-speech, speech-to-text, text-to-text, and image-to-text conversion. These models serve as the building blocks upon which PragYantra's architecture is built, with additional capabilities and concurrency seamlessly integrated to enhance overall performance and user experience.

## So...what does PragYantra mean?

PragYantra, derived from Sanskrit, is a fusion of two words: "Prag" meaning intelligent or wise, and "Yantra" referring to machine or robot. So, put together, PragYantra embodies the concept of an intelligent machine, reflecting the project's goal of creating a flexible platform for experimenting with AI and human-machine interaction.

## Installation

To install and run PragYantra on your device, follow these steps:

Clone the repository:

```bash
git clone https://github.com/sri0606/pragYantra.git
```

Navigate to the project directory and Install dependencies:

```
cd src
pip install -r requirements.txt
```
