from transformers import XCLIPProcessor, XCLIPModel
import torch

model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

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

def run_inference(video, text_categories=text_categories):
    inputs = processor(
    text=text_categories,
    videos=list(video),
    return_tensors="pt",
    padding=True,
)
    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
    probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print(probs)

    max_prob_index = probs.argmax(dim=1).item()
    
    result = text_categories[max_prob_index]
    
    # with longer video, run sliding window and aggregate timestamps per action
    # function will return dic[video_name] = {action: timestamp, action2: timestamp2}
    timestamps = [0]
    
    return result, timestamps

