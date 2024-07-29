from app.models import Video
from app import db

def save_to_database(result, path, embedding):
    video = Video(
        name=path,
        start_time=0,
        end_time=0,
        embedding=embedding
    )
    db.session.add(video)
    db.session.commit()

def query_database():
    return Video.query.all()