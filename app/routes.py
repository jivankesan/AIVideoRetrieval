from flask import current_app as app, render_template, request, redirect, url_for
from . import db
from .models import Video
from scripts.video_processing import process_video
from scripts.inference import run_inference
from scripts.database import save_to_database, query_database
from scripts.embeddings import generate_text_embedding, search_similar_videos

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    if file:
        video_path = f"data/raw_videos/{file.filename}"
        file.save(video_path)
        segments = process_video(video_path)
        inference_results = run_inference(segments)
        query = "This is a video of a person " + str(inference_results)
        embeddings = generate_text_embedding(query)
        save_to_database(inference_results, video_path, embeddings)
        return redirect(url_for('results', filename=file.filename))
    return redirect(url_for('index'))

@app.route('/results')
def results():
    videos = query_database()
    filename = request.args.get('filename')
    return render_template('results.html', videos=videos, filename=filename)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    text_embedding = generate_text_embedding(query)
    results = search_similar_videos(text_embedding)
    return render_template('results.html', videos=results)