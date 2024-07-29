from sentence_transformers import SentenceTransformer, util
from scripts.database import query_database
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_text_embedding(query):
    return model.encode(query, convert_to_tensor=True)

def search_similar_videos(text_embedding):
    videos = query_database()
    video_embeddings = [video.embedding for video in videos]
    cos_scores = util.pytorch_cos_sim(text_embedding, video_embeddings)
    top_results = torch.topk(cos_scores, k=5)
    return [videos[i] for i in top_results.indices]