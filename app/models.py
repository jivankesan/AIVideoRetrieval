from . import db

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    embedding = db.Column(db.PickleType, nullable=False)

    def __repr__(self) -> str:
        return f'<Video {self.name}>'