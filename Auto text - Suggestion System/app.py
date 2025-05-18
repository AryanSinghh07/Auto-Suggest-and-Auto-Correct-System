from flask import Flask, render_template, request
import pandas as pd
import textdistance
import re
from collections import Counter

app = Flask(__name__)

# Load and preprocess corpus
words = []
with open('autocorrect book.txt', 'r', encoding='utf-8') as f:
    data = f.read().lower()
    words = re.findall(r'\w+', data)
    words += words  # Double for frequency bias

# Vocabulary and frequency
words_freq_dict = Counter(words)
total_words = sum(words_freq_dict.values())
probabilities = {word: count / total_words for word, count in words_freq_dict.items()}

@app.route('/')
def index():
    return render_template('index.html', suggestions=None)

@app.route('/suggest', methods=['POST'])
def suggest():
    keyword = request.form['keyword'].lower()
    if keyword:
        # Calculate Jaccard similarity
        similarities = [1 - textdistance.Jaccard(qval=2).distance(vocab_word, keyword) for vocab_word in words_freq_dict.keys()]
        
        df = pd.DataFrame(list(probabilities.items()), columns=['Word', 'Prob'])
        df['Similarity'] = similarities
        
        # Sort by similarity and probability
        suggestions = df.sort_values(['Similarity', 'Prob'], ascending=False).head(10)[['Word', 'Similarity']]
        suggestions_list = suggestions.to_dict('records')
        
        return render_template('index.html', suggestions=suggestions_list)

    return render_template('index.html', suggestions=[])

if __name__ == '__main__':
    app.run(debug=True)
