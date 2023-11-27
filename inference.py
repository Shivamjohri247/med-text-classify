from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import torch

app = FastAPI()

# Serve static files from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

label_dict = {'nervous system diseases': 0,
 'general pathological conditions': 1,
 'cardiovascular diseases': 2,
 'digestive system diseases': 3,
 'neoplasms': 4}

#setting device as cpu for inferencing
device = 'cpu:0'

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model.to(device)
#loading weights
model.load_state_dict(torch.load('model_artifacts/newmodel-epoch-2.model', map_location=torch.device('cpu')))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the HTML form
html_form_path = "static/webpage.html"

@app.post("/predict")
async def predict(inputText: str = Form(...)):
    encoded_data_pred = tokenizer.encode_plus(
        inputText, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=512, 
        return_tensors='pt'
    )

    input_ids_pred = encoded_data_pred['input_ids'].to(device)
    attention_masks_pred = encoded_data_pred['attention_mask'].to(device)
    dataset_test = TensorDataset(input_ids_pred, attention_masks_pred)

    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=1)

    model.eval()
    predictions = []
    
    for batch in dataloader_test:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}

        with torch.no_grad():        
            outputs = model(**inputs)
            logits = outputs[0]
            _, prediction = torch.max(logits, dim=1)
            predictions.append(prediction.item())

    label_dict_inverse = {v: k for k, v in label_dict.items()}
    predicted_class = label_dict_inverse[predictions[0]]

    return {"class": predicted_class, "inputText": inputText}

@app.get("/")
async def read_root():
    return FileResponse(html_form_path)
