# Bert를 이용, 유튜브의 '가장 많이 리플레이된 구간' 그래프를 예측하는 모델 생성
# Prediction of Most-replayed graph by Bert
  
Youtube Most-replayed graph prediction  
using pre-trained Bert model ['bert-base-uncased'](https://huggingface.co/bert-base-uncased)

<br>

## Requirements
- OS : windows 64bit
- Python version : 3.11
- Package list  
```
scikit-learn              1.3.0
lxml                      4.9.3
matplotlib                3.7.2
numpy                     1.24.3
pandas                    2.0.3
pytorch                   1.10.2
selenium                  4.16.0
wandb                     0.16.1
youtube-transcript-api    0.6.1
transformers              4.32.1 
```

<br>

## Project structure
```
Open-source-AI
├─ chrome-win64 # 테스트용 chrome.exe가 들어있는 폴더
├─ chromedriver-win64 # 크롬드라이버가 들어있는 폴더
├─ .gitattributes
├─ all_graph.py # Most-replayed graph를 그리는 코드
├─ baseline_deeper_epoch_20.py # Model 3 epoch 20 
├─ baseline_deeper_epoch_5.py # Model 3 epoch 5
├─ baseline_epoch_20.py # Model 1 epoch 20
├─ baseline_epoch_5.py # Model 1 epoch 5
├─ bert_regression # Model 2 
├─ bfg-1.14.0.jar
├─ get-pip.py 
└─ youtube_url_to_heatmap_coordinates.py # dataset의 라벨 데이터 크롤링 코드
```
**get-pip.py**  
- selenium 설치 시 다음과 같은 에러를 해결하기 위한 파일   
[notice] A new release of pip is available: 23.2.1 -> 23.3.2   
[notice] To update, run: python.exe -m pip install --upgrade pip

=> ```python get-pip.py``` 실행하여 에러 해결 가능

**chrome-win64 폴더**
- 테스트용 chrome.exe가 들어있는 폴더
- 100MB가 넘는 파일 (chrome.dll)이 있어, lfs를 통해 깃허브에 업로드하였음
- 개별 로컬 pc에 설치되어있는 크롬 버전이 다르거나 아예 크롬이 설치되어 있지 않을때 발생 가능한 문제를 없애기 위해 포함
- 버전 : 119.0.6045.105

**chromedriver-win64**
- 크롬드라이버가 들어있는 폴더
- chrome.exe의 버전과 일치해야 에러가 발생하지 않음
- 버전 : 119.0.6045.105
     
**.gitattributes**
- 깃허브 업로드제한 100MB를 넘는 파일 업로드를 위해 **lfs**를 사용하면서 추가
- lfs로 관리할 파일 경로 / 관리옵션 세팅정보가 들어있음
      
**bfg-1.14.0.jar**
- lfs사용시 이미 commit완료하고 push대기중인 100MB이상 파일때문에 발생하는 문제 해결을 위한 파일
- 다운로드 링크 : https://rtyley.github.io/bfg-repo-cleaner/
- lfs 참고링크 : https://newsight.tistory.com/330
- bfg 참고링크 : https://velog.io/@yoogail/%EB%8C%80%EC%9A%A9%EB%9F%89-%ED%8C%8C%EC%9D%BC-github%EC%97%90-push%ED%95%A0-%EB%95%8C-%EC%83%9D%EA%B8%B0%EB%8A%94-%EC%98%A4%EB%A5%98-%EC%A0%95%EB%B3%B5%ED%95%98%EA%B8%B0feat.-git-lfs-bfg

<br>

## How to run (baseline_epoch_5.py)
1. Import libraries
```
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_url_to_heatmap_coordinates import youtube_url_to_heatmap_coordinates
from all_graph import basic_graph, obvious_graph, target_graph, comparison_graph

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
from torch.nn.utils.clip_grad import clip_grad_norm
from transformers import AdamW
import wandb

from datetime import datetime

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
```

2. Data load
```
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
URL = "https://www.youtube.com/watch?v=X7158uQk1yI"
datasets = data_load(URL)
```

3. Tokenization
```
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded_corpus = tokenizer(text=datasets.text.tolist(),
                           add_special_tokens=True,
                           padding='max_length',
                           truncation='longest_first',
                           max_length=300,
                           return_attention_mask=True)
input_ids = encoded_corpus['input_ids']
attention_mask = encoded_corpus['attention_mask']
```

4. Preprocessing
```
def filter_long_texts(tokenizer, texts, max_len):
    indices = []
    lengths = tokenizer(texts, padding=False,
                     truncation=False, return_length=True)['length']
    for i in range(len(texts)):
        if lengths[i] <= max_len-2:
            indices.append(i)
    return indices
short_descriptions = filter_long_texts(tokenizer, datasets.text.tolist(), 300)
input_ids = np.array(input_ids)[short_descriptions]
attention_mask = np.array(attention_mask)[short_descriptions]
labels = datasets.score.to_numpy()[short_descriptions]
```

5. Dataset Split and Preprocessing
```
def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor,
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True)
    return dataloader
test_size = 0.1
seed = 42
batch_size = 32
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
```

6. Define args
```
epoch = 5
config = AutoConfig.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.parameters(),
                        lr=5e-5,
                        eps=1e-8)
loss_function = nn.MSELoss()

total_steps = len(train_dataloader) * epochs
# decreasing learning rate linearly
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, num_training_steps=total_steps)
```

7. Model training
```
class BERTRegressor(nn.Module):
    def __init__(self, config, drop_rate=0.2):
        super(BERTRegressor, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs


def train(model, optimizer, scheduler, loss_function, epochs,
          train_dataloader, test_dataloader, device, clip_value=2):
    for epoch in range(epochs):
        costs = 0
        bn = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            bn += 1

            batch_inputs, batch_masks, batch_labels = tuple(b.to('cpu') for b in batch)

            model.zero_grad()

            outputs = model(batch_inputs, batch_masks)

            loss = loss_function(outputs.squeeze().to(torch.float32),
                                 batch_labels.squeeze().to(torch.float32))

            loss.backward()
            clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()

            costs += loss.item()

            print('Epoch: {}, Batch: {}, Cost: {:.10f}'.format(epoch, step, loss.item()))
            wandb.log({'train/batch_cost': loss.item()})
        print('Epoch: {}, Cost: {}'.format(epoch, costs / bn))
        wandb.log({'train/epoch_cost': costs / bn})

        costs = 0
        model.eval()
        with torch.no_grad():
            bn = 0
            for step, batch in enumerate(test_dataloader):
                bn += 1

                batch_inputs, batch_masks, batch_labels = tuple(b.to('cpu') for b in batch)

                outputs = model(batch_inputs, batch_masks)

                loss = loss_function(outputs, batch_labels)

                costs += loss.item()

            print('Test loss: {}'.format(costs / bn))
            wandb.log({'test/epoch_cost': costs / bn})

        torch.save(model.state_dict(), './model/bert_baseline_epoch_5.pt') # model폴더를 생성 후, 모델에 맞게 이름 설정
    wandb.finish()
    return model
model = BERTRegressor(config=config, drop_rate=0.2)
model = train(model, optimizer, scheduler, loss_function, epochs,
                  train_dataloader, test_dataloader, device, clip_value=2)
```

8. Prediction
```
def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = tuple(b.to('cpu') for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, batch_masks).view(1,-1).tolist()[0]
    return output
testURL = "https://www.youtube.com/watch?v=L6sbfskaTDQ"

val_sets = data_load(testURL)
target = val_sets.score.tolist()
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
```

9. Result
```
## all_graph.py에서
## 모든 plt.savefig(f'./baseline_deeper_20_graph/obvious-graph.png', dpi=300)부분
## 그래프 이미지를 저장할 폴더를 생성하고 폴더명, 파일명을 디렉토리에 맞게, 모델에 맞게 수정하는 것을 권장
basic_graph(y_pred)
obvious_graph(y_pred)
target_graph(target)
comparison_graph(y_pred, target)
print(y_pred)
print('======================================================')
print('Total Length of test dataset:', val_sets.shape[0])
print('Most-replayed point:', y_pred.index(max(y_pred)))

mae = mean_absolute_error(y_test, y_pred)
mdae = median_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mdape = ((pd.Series(y_test) - pd.Series(y_pred)) \
         / pd.Series(y_test)).abs().median()
r_squared = r2_score(y_test, y_pred)

print('MAE: {}, MDAE: {}, MSE: {}, MAPE:{}, '
      'MDAPE: {}, R_squared: {}'.format(mae, mdae, mse, mape, mdape, r_squared))
```
