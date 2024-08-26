# LCC_project
Language, Computation and Cognition Final Project

## Cognitive Task Analysis

This repository contains code and resources for analyzing structured, semistructured, and open-ended cognitive tasks using fMRI and EEG data. The project is divided into two main parts: structured and semistructured tasks, and open-ended tasks.

## Authors

Carmel Soceanu

Peleg Michael

## Repository Structure

- **Structured_and_Semistructured_Tasks.ipynb**: This Jupyter Notebook contains the analysis of structured and semistructured cognitive tasks. It includes data processing, model implementation, and results visualization for tasks that follow predefined structures.

- **Open_Ended_Task/**: This folder contains all the necessary scripts and resources for analyzing open-ended cognitive tasks. The approach here is more flexible and exploratory, allowing for varied and complex responses.

  - **model_embeddings.ipynb**: This notebook creates embedding vectors for the different sentences and words data used in the open-ended cognitive tasks.
  
  - **Save Data.ipynb**: This notebook processes fMRI data by selecting a random subset of voxels for multiple participants across different experiments (Mitchell, Pereira, Toffolo) and saves the processed data into `.pkl` files. It uses the following Python modules:
    - `Mitchell.py`
    - `Pereira.py`
    - `Toffolo.py`
  
  - **Save Graphs.ipynb**: This notebook generates various graphs using the data processed in the project, leveraging utility scripts and results from the analyses. It uses the following Python modules:
    - `Utils.py`
    - `Results.py`
    - `Graphs.py`

## Data Sources

The datasets used in this project are sourced from various studies on cognitive tasks:

- **fMRI Data**: 
  - Mitchell et al., 2008
  - Pereira et al., 2018

- **EEG Data**:
  - Toffolo et al., 2022

## Methodology

The methodologies and neural architectures implemented in this project are inspired by the following works:

- **Neural Architecture**: Inspired by Oota et al., 2018, and Hollenstein et al., 2019
- **General Methodology**: Inspired by Hollenstein et al., 2019
