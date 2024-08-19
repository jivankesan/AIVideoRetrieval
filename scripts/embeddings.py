from sentence_transformers import SentenceTransformer, util
from scripts.database import query_database
import torch
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_text_embedding(query):
    return model.encode(query, convert_to_tensor=True)

def search_similar_videos(text_embedding):
    videos = query_database()
    video_embeddings = []
    valid_videos = []
    for video in videos:
        try:
            embedding_np = np.frombuffer(video.embedding, dtype=np.float32)
            print(video.name)
            if embedding_np.size > 0:
                video_embeddings.append(torch.tensor(embedding_np))
                valid_videos.append(video.name)
        except ValueError as e:
            print(f"Error processing video {video.id}: {e}")
    if not video_embeddings:
        return []
    video_embeddings = torch.stack(video_embeddings)
    if not isinstance(text_embedding, torch.Tensor):
        text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
    
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    text_embedding = text_embedding.to(device)
    video_embeddings = video_embeddings.to(device)
    
    cos_scores = util.pytorch_cos_sim(text_embedding, video_embeddings)

    top_result = torch.argmax(cos_scores).item() 
    
    print(f"this is the number 1 video {valid_videos[top_result]}")
    
    return valid_videos[top_result]

if __name__ == "__main__":
    query = "I am 6 years old"
    res = generate_text_embedding(query)
    print(res)
    print(res.dtype)
    print(res.shape)
    
    query2 = "How old are you young child?"
    res2 = generate_text_embedding(query2)
    print(res2)
    print(res2.dtype)
    print(res2.shape)
    
    cos_scores = util.pytorch_cos_sim(res, res2)
    print(cos_scores)
