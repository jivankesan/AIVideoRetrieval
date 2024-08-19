import io
from . import db
import torch
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import event

Base = declarative_base()

from . import db

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    action = db.Column(db.String(50), nullable=False)  # Change to String to store action name
    embedding = db.Column(db.LargeBinary, nullable=False)

class ActionCount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(50), nullable=False)
    count = db.Column(db.Integer, nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    timestamps = db.Column(db.PickleType, nullable=False)  # Store a list of timestamps
    video = db.relationship('Video', back_populates='actions')

Video.actions = db.relationship('ActionCount', order_by=ActionCount.id, back_populates='video')
