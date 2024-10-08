from flask import Flask, request, jsonify, render_template
import summarizer  # Your summarization module

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    num_sentences = int(data['num_sentences'])
    method = data['method']
    summary = summarizer.summarize_text(text, method=method, num_sentences=num_sentences)
    
    # Assuming summary is returned as a single string, split into list of sentences
    summary_list = summary.split('. ')
    
    return jsonify({'summary': summary_list})

if __name__ == "__main__":
   
    app.run(host='0.0.0.0', port=5000, debug=True)
