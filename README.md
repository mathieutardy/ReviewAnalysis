# Analysis of Reviews

## Context

This is a case study for a company called Center Parcs Europe. They have a network of 22 resorts in Europe. I performed an analysis (topic extraction and sentiment analysis) on customer reviews on TripAdvisor.

## Setup

Go into the highest directory (ReviewAnalysis), then install the dependencies with the following command:

`pip3 install -r requirements.txt`

Then execute the following command:

`python3 -m spacy download en_core_web_sm`

To execute the analysis, run:

`python3 main.py`

## The workflow is the following:

- We retrieve the json dataset.
- Convert the reviews into sentences.
- Compute dictionaries to retrieve topics, sentiment opinions for each sentence.
- Aggregate dictionaries by topics.
- Print results for analysis.

## What to find in the folder src:
- preprocessing.py : transform initial dataset to sentence based reviews.
- topic_sentiment_extraction.py: performs aspect based sentiment analysis on sentences and prints results of largest topics.
- other: contains two exploratory notebooks where I tested two things.
    1. Unsupervised clustering of reviews. First, I used bert to convert reviews into vectors. Then, applied KMean to find clusters.
    2. Used word2vec to find topics within reviews. Then, performed sentiment analysis on the sentences which include the topics using Vader sentiment analysis. Finally, applied LDA to dig into negative reviews of a specific topic.