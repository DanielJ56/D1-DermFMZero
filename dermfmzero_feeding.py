import sys, torch
from PIL import Image
sys.path.insert(0, "src")            # use the bundled open_clip fork
import open_clip
import ast
import pandas as pd
from tqdm import tqdm
from prompts import LISTED_CONDITIONS 

IMAGE_PATH = "C:/Users/danie/Desktop/JoyProject/Main_v2/Images" #Change to Local Image Path

def check_results(ai_answer, derm_labels) -> bool:
    
    response = ai_answer.strip().lower()
    derm_conditions = ast.literal_eval(derm_labels)

    result = response in [cond.strip().lower() for cond in derm_conditions]
    return result

df = pd.read_csv("Master_Results.csv")

if ("dermfmzero_is_correct" not in df.columns) and ("dermfmzero_diagnosis" not in df.columns):
    df["dermfmzero_is_correct"] = None
    df["dermfmzero_diagnosis"] = None

model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:redlessone/DermFM-Zero")
tokenizer = open_clip.get_tokenizer("hf-hub:redlessone/DermFM-Zero")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

with torch.no_grad():
    text = tokenizer([f"this is a dermatology photo of the condition {c}" for c in LISTED_CONDITIONS])
    text_features  = model.encode_text(text.to(device))
    text_features  /= text_features.norm(dim=-1, keepdim=True)

for i, row in tqdm(df.iterrows(), total=len(df), desc="DermFM-Zero"):
    if pd.notna(row["dermfmzero_diagnosis"]):
        continue
    image_name = row["image_path"]
    image = preprocess(Image.open(IMAGE_PATH + "/" + image_name)).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    final_diagnosis = LISTED_CONDITIONS[probs.argmax().item()]
    correctness = check_results(final_diagnosis, row["dermatologist_skin_condition_on_label_name"])

    df.loc[i, "dermfmzero_diagnosis"] = final_diagnosis
    df.loc[i,"dermfmzero_is_correct"] = correctness
    df.to_csv("Master_Results.csv", index=False) #Change filename if do not want to resume upon running

