from flask import current_app as app, render_template, request, redirect, url_for
from . import db
from .models import Video, ActionCount
from scripts.video_processing import process_video
from scripts.inference import run_inference
from scripts.database import save_to_database, query_database, aggregate_action_data
from scripts.embeddings import generate_text_embedding, search_similar_videos

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    if file:
        video_path = f"app/static/cs/{file.filename}"
        file.save(video_path)
        segments = process_video(video_path)
        inference_results, timestamps = run_inference(segments)
        q = "This is a video of a person " + str(inference_results)
        embeddings = generate_text_embedding(q)
        save_to_database(inference_results, video_path,0, embeddings, inference_results, timestamps)
        return redirect(url_for('results', filename=file.filename))
    return redirect(url_for('index'))

@app.route('/results')
def results():
    video_name = request.args.get('video_name')
    video = Video.query.filter_by(name=video_name).first()
    
    # Aggregate action data across all videos
    total_action_counts, action_timestamps = aggregate_action_data()

    return render_template('results.html', filename=video_name, 
                           total_actions=total_action_counts, 
                           action_timestamps=action_timestamps)

@app.route('/search', methods=['GET'])
def search_page():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    text_embedding = generate_text_embedding(query)
    results = search_similar_videos(text_embedding)
    print(results)
    return render_template('search_results.html', video_name=results)

@app.route('/dashboard')
def dashboard():
    # Query the database for total counts of each action
    action_totals = db.session.query(
        ActionCount.action, db.func.sum(ActionCount.count)
    ).group_by(ActionCount.action).all()

    # Convert the query result into a dictionary
    action_data = {action: count for action, count in action_totals}

    return render_template('dashboard.html', action_data=action_data)

@app.route('/delete_video/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    # Query the video by its ID
    video = Video.query.get_or_404(video_id)

    # Delete all associated action counts
    ActionCount.query.filter_by(video_id=video_id).delete()

    # Delete the video entry itself
    db.session.delete(video)

    # Commit the changes to the database
    db.session.commit()

    # Redirect to the dashboard or another page after deletion
    return redirect(url_for('dashboard'))

@app.route('/view_all_videos')
def view_all_videos():
    videos = Video.query.all()
    return render_template('view_all_videos.html', videos=videos)