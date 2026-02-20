# Deep Learning Lab 2: Movie Text Predictions Using GloVe

**Author:** Nimit Mistry  
**ID:** 202511041 
**Course:** Deep Learning  

## Project Overview
This project demonstrates the use of pretrained **GloVe Word Embeddings** (100-Dimensional) alongside standard text-processing techniques (TF-IDF) to train neural network architectures for complex predictive tasks using purely text data. 

We utilize a dataset of movies (`movies.csv`) to predict two distinct targets based on isolated text columns (`overview` and `tagline`):
1. **Model A (Regression):** Predicting the continuous `voting_average` of a movie.
2. **Model B (Multi-Label Classification):** Predicting the various independent `genres` of a movie (e.g., Action, Sci-Fi).

We also perform comprehensive text frequency analysis and apply Logistic Regression mapping on TF-IDF vectors to isolate the purest "Genre-Indicative" words across the dataset.

## Repository Contents
- **`lab2.ipynb`**: The fully runnable Jupyter Notebook containing all data preparation, TF-IDF / GloVe embedding pipelines, PyTorch neural networks, evaluation metrics, and text analysis block outputs.
- **`movies.csv`**: The dataset utilized for all training and evaluation.
- **`README.md`**: Project details and results summary.

## Experimental Results Summary

We compared the predictive performance of identical neural networks when fed purely the `overview` (detailed plot summaries) vs the `tagline` (brief marketing slogans).

### Model A: Rating Prediction (Regression)
- **Baseline Model (Global Mean):** RMSE of ~1.00+
- **Input Column: `Overview`**: Outperformed the baseline. Due to the high word count, the neural network was consistently able to draw weak but positive correlations between specific plot constructs and expected rating distributions.
- **Input Column: `Tagline`**: Consistently performed worse than `Overview`. Marketing taglines are inherently short and generic, failing to provide enough unique linguistic data for the 100D embedding to map onto a reliable continuous rating. 

### Model B: Genre Prediction (Multi-Label Classification)
- **Input Column: `Overview`**: Significantly better. High Macro-F1 and a superb Hamming Loss. The detailed plot summaries explicitly contain words like "alien," "police," or "magic" which perfectly correlate to specific genres like Sci-Fi, Action, or Fantasy.
- **Input Column: `Tagline`**: Performed adequately due to aggressive catchphrases, but severely lacked the depth required to accurately identify multiple secondary sub-genres for a single movie. 

## Task 6 Interpretations 
While basic frequency counting found that narrative pronouns/nouns (like "world," "man," "life") are extremely popular across *all* genres, applying TF-IDF weights mapped via Logistic Regression successfully identified highly isolated indicative words. For example, in the "Action" genre, words representing direct violence or law enforcement carried the highest mathematical weight, correctly modeling how the human brain flags a movie's genre based on keywords.
