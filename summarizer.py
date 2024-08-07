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

def spacy_summarize(text, num_sentences=3):
    doc = nlp(text)
    tokens = []
    stopwords = list(STOP_WORDS)
    allowed_pos = ['ADJ', 'PROPN', 'VERB', 'NOUN']
    for token in doc:
        if token.text in stopwords or token.text in punctuation:
            continue
        if token.pos_ in allowed_pos:
            tokens.append(token.text)

    word_freq = Counter(tokens)
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    sent_tokens = [sent for sent in doc.sents]
    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text.lower()]
                else:
                    sent_scores[sent] += word_freq[word.text.lower()]

    summarized_sentences = nlargest(num_sentences, sent_scores, key=sent_scores.get)
    final_sentences = [sent.text for sent in summarized_sentences]
    summary = " ".join(final_sentences)
    return summary

def transformer_summarize(text, num_sentences=3):
    max_length = min(512, num_sentences * 50)  # Adjust max_length based on desired summary length
    min_length = num_sentences * 10
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def summarize_text(text, method='spacy', num_sentences=3):
    if method == 'spacy':
        return spacy_summarize(text, num_sentences)
    elif method == 'transformer':
        return transformer_summarize(text, num_sentences)
    else:
        raise ValueError("Method must be 'spacy' or 'transformer'")
