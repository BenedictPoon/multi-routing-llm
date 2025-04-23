# MultiLLM Query Router: Getting Started Guide

This guide will help you set up and use the MultiLLM Query Router, an application that intelligently classifies and routes user queries to the appropriate handlers based on content.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Project Structure](#project-structure)
7. [How It Works](#how-it-works)
8. [Customization](#customization)
9. [Troubleshooting](#troubleshooting)

## Overview

The MultiLLM Query Router is a Streamlit-based application that:
- Classifies user input into predefined categories
- Routes queries to appropriate handlers
- Provides relevant responses based on query type
- Tracks conversation history and routing statistics

The system uses LangChain and LangGraph for AI workflows and can utilize different classification methods including LLM-based and rule-based approaches.

## Prerequisites

Before getting started, ensure you have:

- Python 3.8+ installed
- Conda package manager (recommended for environment management)
- An OpenAI API key
- Basic knowledge of Python and LLMs

## Installation

### Step 1: Clone the repository (assuming this is a repo)
```bash
git clone <repository-url>
cd multillm-query-router
```

### Step 2: Create and activate a conda environment
```bash
conda create -n multillm python=3.10
conda activate multillm
```

### Step 3: Install required packages
```bash
pip install -r requirements.txt
```

The requirements include:
- streamlit>=1.27.0
- langchain>=0.0.307
- langchain-core>=0.1.3
- langgraph>=0.0.15
- transformers>=4.35.0
- torch>=2.0.0
- numpy>=1.24.0

## Configuration

### Setting up your OpenAI API Key

You need to set up your OpenAI API key to use the LLM-based classifier. You have two options:

**Option 1: Set as an environment variable (recommended)**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For permanent configuration, add this line to your `~/.bashrc` or `~/.zshrc` file.

**Option 2: Modify the code**
Open `Input Classifier.txt` and uncomment the following line, replacing with your API key:
```python
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Choosing a Classification Method

The project includes two classifier options:
1. **LLM-based classifier** (in `Input Classifier.txt`): Uses OpenAI's GPT models for accurate but token-consuming classification
2. **Rule-based classifier** (in `rule_based_classifier.py.txt`): Uses keyword matching for faster, less accurate classification

To use the rule-based classifier instead of the LLM-based one, you'll need to modify the `router.py` file (which isn't included in the provided files) to import and use the `classify_input` function from `rule_based_classifier.py` instead.

## Running the Application

To start the application:

```bash
python main.py
```

This script will:
1. Check if you're using the correct conda environment ('multillm')
2. Verify that your OpenAI API key is set
3. Launch the Streamlit application

Alternatively, you can run the Streamlit application directly:

```bash
streamlit run ui.py
```

Once running, the application will be accessible in your web browser at `http://localhost:8501`.

## Project Structure

The project consists of the following key files:

- **main.py**: Entry point that checks the environment and launches the app
- **ui.py**: Streamlit user interface code
- **Input Classifier.txt**: LLM-based classifier using OpenAI models
- **rule_based_classifier.py**: Keyword-based classifier (faster alternative)
- **router.py**: (Not provided, but referenced) Contains the logic to process queries based on classification
- **requirements.txt**: Required Python packages

## How It Works

1. **Query Input**: User enters a question or request in the Streamlit interface
2. **Classification**: The query is classified into one of five categories:
   - Technical Question
   - Product Inquiry  
   - Customer Support
   - General Inquiry
   - Other
3. **Routing**: Based on the classification, the query is routed to an appropriate handler
4. **Response Generation**: A relevant response is generated and displayed to the user
5. **History Tracking**: The interaction is added to the conversation history
6. **Statistics**: Query routing statistics are displayed in the sidebar

## Customization

### Adding New Categories

To add new classification categories:

1. Modify the `CATEGORIES` list in `Input Classifier.txt`:
```python
CATEGORIES = ["technical_question", "product_inquiry", "customer_support", "general_inquiry", "other", "your_new_category"]
```

2. If using the rule-based classifier, add keywords for the new category in `rule_based_classifier.py`.

### Changing the UI

The Streamlit UI can be customized by modifying the CSS styles in `ui.py`. Look for the `st.markdown("""<style>...</style>""")` section.

## Troubleshooting

### Common Issues

**Issue**: "OpenAI API key not set" error
**Solution**: Make sure you've set your API key as described in the Configuration section.

**Issue**: "Not in the correct conda environment" warning
**Solution**: Activate the multillm environment with `conda activate multillm`.

**Issue**: Streamlit not found
**Solution**: Make sure you've installed all requirements with `pip install -r requirements.txt`.

### Getting Help

If you continue to experience issues, check the following resources:
- Streamlit documentation: https://docs.streamlit.io/
- LangChain documentation: https://python.langchain.com/docs/
- OpenAI API documentation: https://platform.openai.com/docs/

## Next Steps

Now that you have the MultiLLM Query Router up and running, you might want to:

1. Implement the `router.py` file to fully utilize the system
2. Create specialized handlers for each query category
3. Fine-tune the classification system for your specific use case
4. Expand the UI with additional features

Happy routing!