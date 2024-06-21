# bfsi-sentiment-analysis-winjit-hackathon-
Certainly! Below is the README file for the provided solution.

---

# Sentiment Analysis of Financial News Headlines

## Introduction

This project is developed for the "Gen AI Solutions for BFSI" hackathon. The goal is to predict sentiment for financial news headlines and answer additional questions using Large Language Models (LLMs).

## Problem Statement

Develop a solution to predict sentiment for financial news headlines using natural language processing and Generative AI techniques.

### Data

The training data is provided in a file named `train.xlsx` with the columns [News Headline, Sentiment].

### Primary Task

Train a model on the provided training data to accurately classify sentiment of headlines as positive, negative, or neutral.

### Bonus Tasks

Using LLM and training data, answer the following questions:
1. Find the lowest historical share price for Nokia on days when the headlines had negative sentiment.
2. Determine what field Nokia competes with Google in.

## Solution Overview

The solution involves the following steps:
1. Data Preparation
2. Model Training
3. Streamlit App Development
4. Bonus Tasks using LLM

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/bfsi-sentiment-analysis.git
    cd bfsi-sentiment-analysis
    ```

2. **Install the dependencies**:
    ```bash
    pip install streamlit pandas SpeechRecognition scikit-learn openpyxl openai
    ```

3. **Place the `train.xlsx` file** in the project directory.

4. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## Usage

- **Main Interface**:
  - Enter a news headline manually or use speech input to predict its sentiment.
  - The app will display whether the sentiment is positive, negative, or neutral.

## Project Files

- **app.py**: Main application file containing the Streamlit app code.
- **train.xlsx**: Training data file (to be placed in the project directory).


## Bonus Tasks

- Use an LLM like Google Gemini to answer the following:
  1. Find the lowest historical share price for Nokia on days with negative sentiment headlines.
  2. Determine the field in which Nokia competes with Google.

### Bonus Task 1: Find the Lowest Historical Share Price for Nokia

```python
import openai

# Function to find the lowest share price for Nokia on negative sentiment days
def lowest_nokia_price_on_negative_days(data, lmm_model):
    nokia_data = data[(data['News Headline'].str.contains('Nokia')) & (data['Sentiment'] == 'negative')]
    headlines = nokia_data['News Headline'].tolist()
    
    # Use LLM to find the lowest price
    prompt = f"Find the lowest historical share price for Nokia on days with the following negative headlines: {headlines}"
    response = lmm_model(prompt)
    
    return response

# Example LLM model call
openai.api_key = 'YOUR_API_KEY'
response = openai.Completion.create(
  engine="text-davinci-003",
  prompt="Your prompt here",
  max_tokens=100
)

print(response.choices[0].text.strip())
```

### Bonus Task 2: Determine What Field Nokia Competes with Google In

```python
# Function to find competition field
def nokia_competes_with_google(data, lmm_model):
    nokia_data = data[data['News Headline'].str.contains('Nokia')]
    headlines = nokia_data['News Headline'].tolist()
    
    # Use LLM to determine the field
    prompt = f"Determine what field Nokia competes with Google in based on the following headlines: {headlines}"
    response = lmm_model(prompt)
    
    return response

# Example LLM model call
response = openai.Completion.create(
  engine="text-davinci-003",
  prompt="Your prompt here",
  max_tokens=100
)

print(response.choices[0].text.strip())
```

## Contact

For any queries, please reach out to:
- Name: vaishnavi Matsagar
- Email: vaishno1702@gmail.com
- Name :Mohit Shirvi
- Email: mohitshirvi@gmail.com
---

This README provides a detailed overview of the solution, setup instructions, usage guidelines, and an explanation of the code and bonus tasks.
