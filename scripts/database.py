from .models import Video
from . import db

def save_to_database(results):
    for result in results:
        video = Video(
            name=result['name'],
            start_time=result['start_time'],
            end_time=result['end_time'],
            embedding=result['embedding']
        )
        db.session.add(video)
    db.session.commit()

def query_database():
    return Video.query.all()