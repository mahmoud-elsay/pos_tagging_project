import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the Word2Vec model (you can train it here or load a pre-trained model)
# For the demo, we are assuming that the model is already trained and ready for use.


# Function to preprocess and POS tag the sentence
def text_preprocessing(text):
    stop_words = stopwords.words("english")
    text = text.lower()
    text = re.sub("[^a-zA-z]", " ", text)
    text = re.sub(r"\s+[a-zA-z]\s+", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = word_tokenize(text)
    final_token = [i for i in tokens if i not in stop_words]

    # Lemmatize the tokens
    lemma = WordNetLemmatizer()
    final_words = [lemma.lemmatize(i) for i in final_token if len(i) > 2]
    return final_words


def pos_tag_text(text):
    """
    Apply POS tagging to text and return tagged tokens
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return [f"{word}_{tag}" for word, tag in tagged]


# Title of the web app
st.title("POS Tagging with Streamlit")

# Text input from user
input_text = st.text_area("Enter a sentence to tag:", "Enter your sentence here...")

# When the user clicks the button
if st.button("Tag Sentence"):
    if input_text:
        # Preprocess and POS tag the text
        processed_text = text_preprocessing(input_text)
        pos_tagged = pos_tag_text(input_text)

        # Display POS tagged result
        st.subheader("POS Tagged Output")
        st.write(f"Original Sentence: {input_text}")
        st.write(f"POS Tagged: {', '.join(pos_tagged)}")

        # Visualizing POS tag distribution
        all_tags = [tag.split("_")[1] for tag in pos_tagged]
        tag_counts = Counter(all_tags)

        # Plot POS tag distribution
        st.subheader("POS Tag Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=list(tag_counts.keys()),
            y=list(tag_counts.values()),
            palette="viridis",
            ax=ax,
        )
        ax.set_title("Distribution of POS Tags", fontsize=16)
        ax.set_xlabel("POS Tags", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.error("Please enter a sentence for POS tagging.")
