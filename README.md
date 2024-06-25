# RAGnetic 
```
  ____      _    ____            _   _      
 |  _ \    / \  / ___|_ __   ___| |_(_) ___ 
 | |_) |  / _ \| |  _| '_ \ / _ \ __| |/ __|
 |  _ <  / ___ \ |_| | | | |  __/ |_| | (__ 
 |_| \_\/_/   \_\____|_| |_|\___|\__|_|\___|
                   
                      DSPY RAG 
```


![1707471817175](https://github.com/Kirouane-Ayoub/RAGnetic/assets/99510125/e8b24a4d-9b35-4d0a-aecd-9def301e4d04)


## Overview
This project implements a **Retrieval-Augmented Generation (RAG) system** using **DSPy**. **RAG** allows **Language Models (LLMs)** to leverage a large corpus of knowledge to dynamically retrieve relevant passages and generate well-refined responses. **DSPy** facilitates the setup of **RAG pipelines** by providing tools for optimizing prompts and managing the flow of information between retrieval and generation stages.

## Components Used
- **DSPy Framework**: Used for algorithmically optimizing LLM prompts and weights within the pipeline.
- **Language Model (LLM)**: **Cohere** `command-r-plus` is employed as the primary LLM for generating responses.
- **Embedding Model**: **Cohere** `embed-english-v2.0` is used for generating embeddings.
- **Vector Store Database**: `ChromaDB` serves as the database for storing vector representations.

## Getting Started
1. **Installation**:
   - Clone the repository:
     ```
     git clone https://github.com/Kirouane-Ayoub/RAGnetic.git
     cd RAGnetic
     ```
   - Install dependencies (ensure Python 3.8+):
     ```
     pip install -r requirements.txt
     ```

2. **Setting Up Environment**:
   - Ensure DSPy and necessary models (Cohere `command-r-plus`, `embed-english-v2.0`) are correctly configured and accessible.
   - Configure ChromaDB for storing and retrieving vector data .
   - Modify configurations in `settings.py` as per your environment and requirements.
   
3. **Running the Project**:
   - Execute the main script:
     ```
     python main.py
     ```
