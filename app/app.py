from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertModel, BertTokenizer
import joblib
import uvicorn
import numpy as np

app = FastAPI()

bert_model = BertModel.from_pretrained('../fine_tuned_german_bert',ignore_mismatched_sizes=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased',ignore_mismatched_sizes=True)


bert_model.eval()

# Load the Random Forest model
rf_model = joblib.load('./trained_models/Random_forest_Bert_model.joblib')

class TextData(BaseModel):
    text: str

def generate_embeddings(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=150)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

@app.post("/predict")
async def predict(data: TextData):
    try:
        embeddings = generate_embeddings(data.text)
        prediction = rf_model.predict([embeddings])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=404, detail= str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
