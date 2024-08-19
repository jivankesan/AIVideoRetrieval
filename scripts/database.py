from app.models import Video, ActionCount
import torch
from app import db
import pickle

def save_to_database(result, path, start_time, embedding, action, timestamps):
    video = Video(
        name=path.split("/")[-1],
        start_time=0,
        end_time=0,
        action = action,
        embedding=embedding.detach().cpu().numpy().tobytes()
        )
    db.session.add(video)
    db.session.flush()
    
    action_count = ActionCount(
            action=action,
            count=len(timestamps),
            video_id=video.id,
            timestamps=timestamps
        )
    db.session.add(action_count)
    
    db.session.commit()
    

def query_database():
    videos = Video.query.all()
    return videos

def aggregate_action_data():
    actions = ActionCount.query.all()
    total_action_counts = {}
    action_timestamps = {}

    for action in actions:
        if action.action in total_action_counts:
            total_action_counts[action.action] += action.count
            action_timestamps[action.action].extend(action.timestamps)
        else:
            total_action_counts[action.action] = action.count
            action_timestamps[action.action] = action.timestamps

    return total_action_counts, action_timestamps
