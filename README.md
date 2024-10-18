# CodeAlpha_AI_Music

An AI-powered music generation system capable of composing original music using advanced deep learning techniques. This project utilizes Recurrent Neural Networks (RNNs) to analyze MIDI files and generate new compositions.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Cloning the Project](#cloning-the-project)
- [Installation Requirements](#installation-requirements)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [License](#license)

## Features

- **MIDI Processing**: Load and process MIDI files to extract musical notes.
- **RNN Training**: Train a Recurrent Neural Network on the processed notes to generate new music.
- **Music Generation**: Generate original music compositions in MIDI format.

## Getting Started

To get started with this project, follow the instructions below:

### Cloning the Project

To clone this repository, use the following command:

```bash
git clone https://github.com/Csasaka19/CodeAlpha_AI_Music.git

cd CodeAlpha_AI_Music
```

### Installation Requirements

This project requires Python 3.x and several libraries. You can install the necessary libraries using pip. First, ensure you have pip installed, then run the following command:

```bash
pip install -r requirements.txt
```

Run the project in this manner(training the model might take sometime):

Run src/data_preprocessing.py to extract notes and prepare sequences.
Run src/train_model.py to train the model and save it in model/.
Run src/generate_music.py to generate new music, which will be saved in output/.

 ~ The JupyterNotebooks contains the documentation about them.

### Project Structure
```bash
CodeAlpha_AI_Music/
│
├── data/
│   ├── midi_files/                  # Folder to store your raw MIDI files
│   ├── processed/                   # Folder to store pre-processed data
│
├── model/
│   ├── music_generator_model.h5     # Trained Keras model
│   └── checkpoints/                 # Directory to store model checkpoints (optional)
│
├── notebooks/
│   ├── data_preparation.ipynb       # Jupyter notebook to pre-process MIDI files
│   ├── model_training.ipynb         # Jupyter notebook for model training
│   └── music_generation.ipynb       # Jupyter notebook for generating new music
│
├── output/
│   └── generated_music.mid          # Output folder to save the generated music files
│
├── src/
│   ├── data_preprocessing.py        # Script to handle MIDI processing
│   ├── train_model.py               # Script for training the RNN model
│   └── generate_music.py            # Script to generate new music based on the trained model
│
├── README.md                        # Documentation on how to set up and run the project
├── requirements.txt                 # List of required Python libraries
└── .gitignore                       # Ignore unnecessary files

```


### How to Contribute

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

Fork the repository on GitHub.
Create a new branch for your feature or bug fix:

```bash
git checkout -b feature/your-feature-name
```
Make your changes and commit them:
```bash
git commit -m "Add your message here"
```
Push to your branch:

```bash

git push origin feature/your-feature-name
```
Create a pull request to the main repository.


### License

This project is licensed under the GNU Affero General Public License. See the LICENSE file for more information.
