from flask import Flask, request, render_template
import summarizer  # Import your summarization functions

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    num_sentences = int(request.form['num_sentences'])
    method = request.form['method']
    summary = summarizer.summarize_text(text, method=method, num_sentences=num_sentences)
    return render_template('result.html', original_text=text, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
