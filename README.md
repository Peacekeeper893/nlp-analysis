# Game of Thrones NLP Analysis

This project leverages Natural Language Processing (NLP) techniques to analyze the subtitles of the "Game of Thrones" series. It encompasses two main components:

1. **Character Network Generator**: Identifies and visualizes character interactions.
2. **Theme Classifier**: Detects and visualizes underlying themes within the dialogue.

## Table of Contents

- [Project Overview](#project-overview)
- [Character Network Generator](#character-network-generator)
  - [Background](#background)
  - [Working](#working)
  - [Results](#results)
- [Theme Classification](#theme-classification)
  - [Background](#background-1)
  - [Working](#working-1)
  - [Results](#results-1)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Project Overview

This project performs comprehensive analysis of the "Game of Thrones" series by:

- **Character Network Generator**: Extracting interactions between characters and visualizing them as a network graph.
- **Theme Classification**: Identifying and visualizing themes such as friendship, hope, betrayal, and more within the series' dialogues.


## Character Network Generator

### Background

The **Character Network Generator** analyzes the interactions between characters in the "Game of Thrones" series. By employing Named Entity Recognition (NER), it identifies character mentions and visualizes their relationships as a network graph, highlighting the complexity and dynamics of character interactions throughout the series.

### Working

1. **Setup and Dependencies**

   Utilizes libraries such as `spacy`, `nltk`, `pandas`, `numpy`, `matplotlib`, `networkx`, and `pyvis` for tasks including model loading, data manipulation, and visualization.

2. **Named Entity Recognition (NER)**

   Uses the `spacy` library with the `en_core_web_trf` model to perform NER on the subtitles, extracting character names and their interactions.

3. **Dataset Loading and Preprocessing**

   Reads the subtitles dataset from `GOT.csv` and aggregates sentences per season and episode for analysis.

4. **Character Mapping**

   Maps various aliases of characters to ensure consistent identification across different mentions (e.g., "Daenerys Targaryen" as "daenerys").

5. **NER Inference**

   Applies the NER model to extract character names from the subtitles.

6. **Generate Character Network**

   Creates a network graph of character interactions using `networkx` and visualizes it with `pyvis`.


### Results

Refer to the `character-network` directory and the readme files for the character network results and visualizations.

## Theme Classification

### Background

The **Theme Classification** component analyzes the subtitles of the "Game of Thrones" series to identify and visualize underlying themes such as friendship, hope, betrayal, and more. Utilizing zero-shot classification models, it provides insights into the prevalent themes within each episode's dialogue.

### Working

1. **Setup and Dependencies**

   Uses libraries like `transformers`, `nltk`, `pandas`, `numpy`, `torch`, `matplotlib`, and `seaborn` for model loading, data manipulation, and visualization.

2. **Model Loading**

   Employs the `facebook/bart-large-mnli` model for zero-shot classification, configured to run on a GPU if available.

3. **Dataset Loading and Preprocessing**

   Loads the subtitles dataset from `GOT.csv`, processing it to aggregate sentences per season and episode.

4. **Theme Inference**

   Defines a list of themes to classify, processes the subtitles in batches, applies the theme classifier, and aggregates the scores for each theme.

5. **Visualization**

   Generates bar plots to visualize the scores of each identified theme.


### Results

Refer to the `theme-classifier` directory and the readme files for the theme classification results and visualizations.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/got-nlp-analysis.git
    cd got-nlp-analysis
    ```

2. **Set Up Virtual Environment**

    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On Unix or MacOS
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    *Ensure you have a `requirements.txt` file at the root of the project listing all necessary packages.*

## Usage

### Character Network Generator

1. Navigate to the `character-network` directory:

    ```bash
    cd character-network
    ```

2. Open the Jupyter Notebook and run the cells:

    ```bash
    jupyter notebook character_network_generator.ipynb
    ```

3. View the generated network graph in the `results` folder.

### Theme Classification

1. Navigate to the `theme-classifier` directory:

    ```bash
    cd theme-classifier
    ```

2. Open the Jupyter Notebook and run the cells:

    ```bash
    jupyter notebook theme_classification_development.ipynb
    ```

3. View the theme classification results and visualizations.

## Requirements

- Python 3.7+
- Transformers
- NLTK
- Pandas
- NumPy
- Torch
- Matplotlib
- Seaborn
- Spacy
- NetworkX
- Pyvis

*Additional requirements may be listed in the `requirements.txt` file.*

