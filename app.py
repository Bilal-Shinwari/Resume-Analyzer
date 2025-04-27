from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
import docx
import PyPDF2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text(file_storage):
    filename = file_storage.filename.lower()
    text = ""

    if filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file_storage)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    elif filename.endswith('.docx'):
        doc = docx.Document(file_storage)
        for para in doc.paragraphs:
            text += para.text + '\n'
    elif filename.endswith('.txt'):
        text = file_storage.read().decode('utf-8')

    return text.strip()

def compute_similarity(job_desc, resume_text):
    job_emb = model.encode([job_desc])[0]
    resume_emb = model.encode([resume_text])[0]
    score = util.cos_sim(job_emb, resume_emb).item()
    return round(score * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_desc = request.form['job_description']
        cv_file = request.files['cv_file']

        if job_desc and cv_file:
            filename = secure_filename(cv_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv_file.save(filepath)

            # Reopen in binary mode just to use for upload, but for parsing use the original cv_file
            cv_file.stream.seek(0)  # Reset pointer to the beginning
            text = extract_text(cv_file)

            score = compute_similarity(job_desc, text)

            result_data = {
                'total_score': score,
                'semantic_score': score,
                'skills_score': round(score * 0.6, 2),
                'experience_score': round(score * 0.3, 2),
                'education_score': round(score * 0.2, 2),
                'cv_text': text[:1000] + '...' if len(text) > 1000 else text,
                'job_description': job_desc
            }
            return render_template('results.html', **result_data)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)

