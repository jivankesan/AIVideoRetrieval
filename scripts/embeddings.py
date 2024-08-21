from sentence_transformers import SentenceTransformer, util
# from scripts.database import query_database
import torch
import numpy as np
import csv

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
    text_categories = [
    "Cook.Cleandishes",        # 1
    "Cook.Cleanup",            # 2
    "Cook.Cut",                # 3
    "Cook.Stir",               # 4
    "Cook.Usestove",           # 5
    "Cutbread",                # 6
    "Drink.Frombottle",        # 7
    "Drink.Fromcan",           # 8
    "Drink.Fromcup",           # 9
    "Drink.Fromglass",         # 10
    "Eat.Attable",             # 11
    "Eat.Snack",               # 12
    "Enter",                   # 13
    "Getup",                   # 14
    "Laydown",                 # 15
    "Leave",                   # 16
    "Makecoffee.Pourgrains",   # 17
    "Makecoffee.Pourwater",    # 18
    "Maketea.Boilwater",       # 19
    "Maketea.Insertteabag",    # 20
    "Pour.Frombottle",         # 21
    "Pour.Fromcan",            # 22
    "Pour.Fromkettle",         # 23
    "Readbook",                # 24
    "Sitdown",                 # 25
    "Takepills",               # 26
    "Uselaptop",               # 27
    "Usetablet",               # 28
    "Usetelephone",            # 29
    "Walk",                    # 30
    "WatchTV"                  # 31
]
    
    with open('text_embeddings.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Text', 'Embedding'])  # Write the header
        
        for text in text_categories:
            embedding = generate_text_embedding(text)
            writer.writerow([text, embedding])


