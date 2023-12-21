from youtube_transcript_api import YouTubeTranscriptApi
from youtube_url_to_heatmap_coordinates import youtube_url_to_heatmap_coordinates
import pandas as pd
import numpy as np
import torch

import torch.nn as nn
from transformers import BertModel
from transformers import AutoTokenizer, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

import wandb

wandb.login(key="f4d64ffeb02e32c7b21d705a5a6316dacd7ffb0f")

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

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

    datasets_col_name = ['text', 'time_point', 'score']
    df_datasets = pd.DataFrame(list_datasets, columns=datasets_col_name)
    return df_datasets


def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor,
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True)
    return dataloader


def filter_long_texts(tokenizer, texts, max_len):
    indices = []
    lengths = tokenizer(texts, padding=False,
                     truncation=False, return_length=True)['length']
    for i in range(len(texts)):
        if lengths[i] <= max_len-2:
            indices.append(i)
    return indices


# Training
class BERTRegressor(nn.Module):
    def __init__(self, config, drop_rate=0.2):
        super(BERTRegressor, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs


def train(model, optimizer, scheduler, loss_function, epochs,
          train_dataloader, test_dataloader, device):
    for epoch in range(epochs):
        train_loss = 0
        bn = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            bn += 1

            optimizer.zero_grad()

            batch_inputs, batch_masks, batch_labels = tuple(b.to('cpu') for b in batch)

            outputs = model(batch_inputs, batch_masks)

            loss = loss_function(outputs.squeeze().to(torch.float32),
                                 batch_labels.squeeze().to(torch.float32))

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            print('Epoch: {}, Batch: {}, Cost: {:.10f}'.format(epoch, step, loss.item()))
            wandb.log({'train/batch_cost': loss.item()})
        print('Epoch: {}, Cost: {}'.format(epoch, train_loss / bn))
        wandb.log({'train/epoch_cost': train_loss / bn})

        model.eval()
        with torch.no_grad():
            val_loss = 0
            bn = 0
            for step, batch in enumerate(test_dataloader):
                bn += 1

                batch_inputs, batch_masks, batch_labels = tuple(b.to('cpu') for b in batch)

                outputs = model(batch_inputs, batch_masks)

                loss = loss_function(outputs.squeeze().to(torch.float32),
                                     batch_labels.squeeze().to(torch.float32))

                val_loss += loss.item()

            print('Test loss: {}'.format(val_loss / bn))
            wandb.log({'test/epoch_cost': val_loss / bn})
        torch.save(model.state_dict(), './bert_regression_deep.pt')

    return model


# Prediction
def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = tuple(b.to('cpu') for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, batch_masks).view(1,-1).tolist()[0]
    return output


# Main
def main():

    wandb.init(project="openai",
               name='bert_for_youtube_most_replayed_graph_deep'
               )

    # Set args
    test_size = 0.1
    seed = 42
    batch_size = 32
    epochs = 50

    # Data load
    URL = "https://www.youtube.com/watch?v=X7158uQk1yI"

    datasets = data_load(URL)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    encoded_corpus = tokenizer(text=datasets.text.tolist(),
                               add_special_tokens=True,
                               padding='max_length',
                               truncation='longest_first',
                               max_length=300,
                               return_attention_mask=True)
    input_ids = encoded_corpus['input_ids']
    attention_mask = encoded_corpus['attention_mask']

    short_descriptions = filter_long_texts(tokenizer, datasets.text.tolist(), 300)
    input_ids = np.array(input_ids)[short_descriptions]
    attention_mask = np.array(attention_mask)[short_descriptions]
    labels = datasets.score.to_numpy()[short_descriptions]

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=test_size,
                                                                            random_state=seed)
    train_masks, test_masks, _, _ = train_test_split(attention_mask,
                                                     labels, test_size=test_size,
                                                     random_state=seed)

    score_scaler = StandardScaler()
    score_scaler.fit(train_labels.reshape(-1, 1))
    train_labels = score_scaler.transform(train_labels.reshape(-1, 1))
    test_labels = score_scaler.transform(test_labels.reshape(-1, 1))

    train_dataloader = create_dataloaders(train_inputs, train_masks,
                                          train_labels, batch_size)
    test_dataloader = create_dataloaders(test_inputs, test_masks,
                                         test_labels, batch_size)

    # train
    config = AutoConfig.from_pretrained('bert-base-uncased')
    model = BERTRegressor(config=config, drop_rate=0.2)
    optimizer = AdamW(model.parameters(),
                            lr=5e-5,
                            eps=1e-8)
    loss_function = nn.MSELoss()

    total_steps = len(train_dataloader) * epochs
    # decreasing learning rate linearly
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, num_training_steps=total_steps)

    model = train(model, optimizer, scheduler, loss_function, epochs,
                  train_dataloader, test_dataloader, device)


    # Prediction
    testURL = "https://www.youtube.com/watch?v=L6sbfskaTDQ"

    val_sets = data_load(testURL)

    encoded_val_corpus = tokenizer(text=val_sets.text.tolist(),
                  add_special_tokens=True,
                  padding='max_length',
                  truncation='longest_first',
                  max_length=300,
                  return_attention_mask=True)
    val_input_ids = np.array(encoded_val_corpus['input_ids'])
    val_attention_mask = np.array(encoded_val_corpus['attention_mask'])
    val_labels = val_sets.score.to_numpy()
    val_labels = score_scaler.transform(val_labels.reshape(-1, 1))
    val_dataloader = create_dataloaders(val_input_ids,
                                        val_attention_mask, val_labels, batch_size)

    y_pred_scaled = predict(model, val_dataloader, device)
    y_test = val_sets.score.to_numpy()
    y_pred = score_scaler.inverse_transform(np.array(y_pred_scaled).reshape(-1,1))
    y_pred = y_pred.reshape(1, -1).tolist()[0]
    print('=======================')
    print('Graph')
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter1d
    print(y_pred)
    x = range(len(y_pred))
    y = y_pred
    y = gaussian_filter1d(y, sigma=12)
    plt.fill_between(x, y, 0,
                     facecolor="grey",  # The fill color
                     color='grey',  # The outline color
                     alpha=0.2)  # Transparency of the fill

    # Show the plot
    # plt.show()
    plt.savefig(f'bert-deep.png', dpi=300)
    print('=======================')
    print('Total Length of test dataset:', val_sets.shape[0])
    print('Most-replayed point:', y_pred.index(max(y_pred)))

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import median_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import r2_score
    mae = mean_absolute_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mdape = ((pd.Series(y_test) - pd.Series(y_pred)) \
             / pd.Series(y_test)).abs().median()
    r_squared = r2_score(y_test, y_pred)

    print('MAE: {}, MDAE: {}, MSE: {}, MAPE:{}, '
          'MDAPE: {}, R_squared: {}'.format(mae, mdae, mse, mape, mdape, r_squared))

if __name__ == '__main__':
    main()
