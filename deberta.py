from youtube_transcript_api import YouTubeTranscriptApi
from youtube_url_to_heatmap_coordinates import youtube_url_to_heatmap_coordinates
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer
import torch.optim as optim
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

# Data load
def data_load(url):
    # transcript
    video_id = url.split("v=")[1]
    transcripts = YouTubeTranscriptApi.get_transcript(video_id)
    # heatmap
    heatmap_list = youtube_url_to_heatmap_coordinates(url)
    col_name = ['point', 'score']
    heatmap_df = pd.DataFrame(heatmap_list, columns=col_name)
    heatmap_df['point'] = round(heatmap_df['point'], 2)

    list_datasets = []
    for transcript in transcripts:
        text = transcript['text']
        point = round(transcript['start'] / transcripts[-1]['start'], 2)
        score = heatmap_df[heatmap_df['point'] == point]['score'].iloc[0]
        list_datasets.append([text, point, score])

    datasets_col_name = ['subtitle', 'time_point', 'score']
    df_datasets = pd.DataFrame(list_datasets, columns=datasets_col_name)
    return df_datasets

test_URL = "https://www.youtube.com/watch?v=X7158uQk1yI"
datasets = data_load(test_URL)
print(datasets)
train_df, valid_df = train_test_split(datasets, test_size=0.25, random_state=21)

# Model
class Subtitle_Dataset(Dataset):

    def __init__(self, data, maxlen, tokenizer):
        # Store the contents of the file in a pandas dataframe
        self.df = data.reset_index()
        # Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        # Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Select the sentence and label at the specified index in the data frame
        subtitle = self.df.loc[index, 'subtitle']
        try:
            target = self.df.loc[index, 'score']
        except:
            target = 0.0
        # Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.tokenize(subtitle)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']
            # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()

        target = torch.tensor(target, dtype=torch.float32)

        return input_ids, attention_mask, target

class BertRegresser(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(config.hidden_size,128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128,128)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(128,1)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        return output

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

## Model Configurations
MAX_LEN_TRAIN = 205
MAX_LEN_VALID = 205
MAX_LEN_TEST = 205
BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 10
NUM_THREADS = 1  ## Number of threads for collecting dataset
MODEL_NAME = 'bert-base-uncased'

config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BertRegresser.from_pretrained(MODEL_NAME, config=config)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=LR)

def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)

            mean_loss += criterion(output, target.type_as(output)).item()
            #             mean_err += get_rmse(output, target)
            count += 1

    return mean_loss / count
# Train
def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_acc = 0
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, target) in enumerate(iterable=train_loader):
            optimizer.zero_grad()

            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(output, target.type_as(output))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Training loss is {train_loss / len(train_loader)}")
        val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
        print("Epoch {} complete! Validation Loss : {}".format(epoch, val_loss))

def main():
    ## Training Dataset
    train_set = Subtitle_Dataset(data=train_df, maxlen=MAX_LEN_TRAIN, tokenizer=tokenizer)
    valid_set = Subtitle_Dataset(data=valid_df, maxlen=MAX_LEN_VALID, tokenizer=tokenizer)

    ## Data Loaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)

    train(model=model,
          criterion=criterion,
          optimizer=optimizer,
          train_loader=train_loader,
          val_loader=valid_loader,
          epochs = NUM_EPOCHS,
          device = device)

if __name__ == '__main__':
    main()
print("")