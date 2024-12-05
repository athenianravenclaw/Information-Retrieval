import os
import re
import nltk
import pickle
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Removing punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenizing and remove stop words
    tokens = [word for word in text.lower().split() if word not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

def build_inverted_index(data_path):
    inverted_index = defaultdict(list)
    
    with open(os.path.join(data_path, 'CISI.ALL'), 'r') as file:
        doc_id = None
        text_block = []
        for line in file:
            if line.startswith('.I'):
                if doc_id is not None:
                    # Process the text block for the previous document
                    text = ' '.join(text_block)
                    tokens = preprocess_text(text)
                    for token in tokens:
                        inverted_index[token].append(doc_id)
                    
                doc_id = line.split()[1]
                text_block = []
            elif line.startswith('.W'):
                text_block.append(next(file).strip())
    
    if doc_id is not None and text_block:
        text = ' '.join(text_block)
        tokens = preprocess_text(text)
        for token in tokens:
            inverted_index[token].append(doc_id)
    vocabulary_length = len(inverted_index)
    print(f"Vocabulary Length: {vocabulary_length}")
    with open(f'model_queries_21EC39023.bin', 'wb') as f:
        pickle.dump(inverted_index, f)
    
    # Printing a portion (first 10) of the inverted index
    print("Sample of the Inverted Index:")
    for key in list(inverted_index)[:10]:  
        print(f"{key}: {inverted_index[key]}")

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]
    build_inverted_index(data_path)
