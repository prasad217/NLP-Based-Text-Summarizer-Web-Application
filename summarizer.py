import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load transformer model
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")


def spacy_summarize(text, num_sentences):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.text not in punctuation and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
    
    word_freq = Counter(tokens)
    max_freq = max(word_freq.values())
    word_freq = {word: freq/max_freq for word, freq in word_freq.items()}
    
    sent_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_freq:
                sent_scores[sent] = sent_scores.get(sent, 0) + word_freq[word.text.lower()]
    
    summarized_sentences = nlargest(num_sentences, sent_scores, key=sent_scores.get)
    summarized_sentences = sorted(summarized_sentences, key=lambda sent: sent.start)
    summary = " ".join([sent.text for sent in summarized_sentences])
    return summary


def transformer_summarize(text, num_sentences=3):
    max_length = min(512, num_sentences * 50)  # Adjust max_length based on desired summary length
    min_length = num_sentences * 20  # Increased min_length for better summaries
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=True)
    return summary[0]['summary_text']

def summarize_text(text, method='spacy', num_sentences=3):
    if method == 'spacy':
        return spacy_summarize(text, num_sentences)
    elif method == 'transformer':
        return transformer_summarize(text, num_sentences)
    else:
        raise ValueError("Method must be 'spacy' or 'transformer'")
