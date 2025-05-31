# Hate-speech-detection

## Project Overview

This project focuses on analyzing and classifying tweets into one of three categories:

1. **Hate Speech**
2. **Offensive Language**
3. **No Hate or Offensive Language**

The main objective is to build a machine learning model capable of accurately identifying the sentiment and tone of tweets while addressing challenges like imbalanced datasets and complex text data.

---

## Dataset

The dataset used in this project is a CSV file named `twitter.csv`, containing the following columns:

* **tweet**: The text content of the tweet.
* **class**: The label associated with the tweet (

  * `0`: Hate Speech
  * `1`: Offensive Language
  * `2`: No Hate or Offensive Language)

To make the dataset more user-friendly, the `class` column was mapped to human-readable labels (`Hate Speech`, `Offensive Language`, and `No Hate or Offensive Language`).

---

## Project Workflow

### 1. **Data Preprocessing**

Text data often contains noise that needs to be removed to ensure better model performance. The following steps were performed:

* **Lowercasing**: Converting all text to lowercase.
* **Removing URLs**: Deleting any web links from the text.
* **Removing HTML Tags and Brackets**: Cleaning extraneous formatting.
* **Removing Punctuation and Numbers**: To focus only on meaningful words.
* **Stopword Removal**: Eliminating common English words that add little value to sentiment analysis.
* **Stemming**: Reducing words to their root forms using the Snowball Stemmer.

The `clean_data` function handles all the preprocessing tasks.

### 2. **Feature Extraction**

To convert the cleaned text into a machine-readable format, the `TfidfVectorizer` was employed:

* **TF-IDF (Term Frequency-Inverse Document Frequency)**: Captures the importance of terms in a document relative to the corpus.
* **N-grams**: Used both unigrams and bigrams (`ngram_range=(1, 2)`) to capture context from adjacent words.
* **Feature Limit**: Limited to the top 5,000 terms to reduce noise and computational overhead.

### 3. **Data Splitting**

The dataset was split into training and testing sets with a 70-30 ratio using `train_test_split` from Scikit-Learn. Stratified sampling (`stratify=y`) was used to ensure the class distribution remained consistent across both sets.

### 4. **Model Building**

A `RandomForestClassifier` was selected for its robustness and ability to handle imbalanced datasets. Key parameters included:

* `n_estimators=200`: Number of trees in the forest.
* `max_depth=15`: Limit tree depth to prevent overfitting.
* `class_weight='balanced'`: Adjusts class weights inversely proportional to class frequencies.

### 5. **Evaluation**

The model's performance was assessed using:

* **Accuracy Score**: Overall proportion of correctly classified tweets.
* **Confusion Matrix**: Visualization of true vs. predicted classifications.
* **Classification Report**: Detailed metrics (precision, recall, F1-score) for each class.

### 6. **Sample Prediction**

The model was tested with sample tweets to verify its ability to classify new, unseen data. Predictions were generated after cleaning and vectorizing the input text.

---

## Dependencies

The following Python libraries are required:

* `pandas`
* `numpy`
* `nltk`
* `scikit-learn`
* `seaborn`
* `matplotlib`

Install these libraries using pip:

```bash
pip install pandas numpy nltk scikit-learn seaborn matplotlib
```

---

## Running the Code

1. Place the `twitter.csv` file in the same directory as the script.
2. Run the script using Python.
3. Check the output for accuracy scores, confusion matrix, and classification reports.

### Example Output

* **Accuracy Score**: Displays the overall accuracy of the model.
* **Confusion Matrix**: Provides insights into misclassifications.
* **Sample Prediction**: Prints the predicted class of a given tweet.

---

## Potential Improvements

1. **Hyperparameter Tuning**:
   Use Grid Search or Random Search to further optimize Random Forest parameters.

2. **Feature Expansion**:
   Experiment with additional feature extraction techniques like Word2Vec or GloVe embeddings.

3. **Advanced Models**:
   Implement state-of-the-art models like BERT or LSTM for better contextual understanding of tweets.

4. **Data Augmentation**:
   Generate synthetic data through back-translation or oversample minority classes.

---

## Conclusion

This project demonstrates the end-to-end process of building a sentiment analysis model for tweets, including preprocessing, feature extraction, model training, and evaluation. Despite challenges like class imbalance, the model performs reasonably well, and further enhancements can yield even better results.
