from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BertTokenizer
from transformers import BertConfig, BertModel
import torch

# URL : https://www.youtube.com/watch?v=X7158uQk1yI
# Title : FULL MATCH: Brazil vs. France 200  v6 FIFA World Cup
transcript = YouTubeTranscriptApi.get_transcript("X7158uQk1yI")
print(transcript)

# sentence_array에 text만 모아서 넣어줌
sentence_array = []
for element in transcript:
    sentence_array.append(element["text"])
print(sentence_array) # 출력결과 : [ "문장1..", "문장2...", "문장3...", ... "문장n..." ]

# BertTokenizer를 이용해 문장들을 토큰화 : bert-base-uncased 모델
features_array = []
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Bert-base의 토크나이저
for sentence in sentence_array:
    print(tokenizer.tokenize(sentence))
    features_array.append(tokenizer(sentence, max_length=17, padding="max_length", truncation=True, return_tensors="pt")) # return_tensors를 "pt"로 주어 피처를 토치 텐서로 변환
print(features_array)


# 모델 초기설정 : bert-base-uncased 로 통일
pretrained_model_config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", config=pretrained_model_config)



# BERT에 태우기
'''
for feature in features_array:
  print(feature)
  outputs = model(**feature)
  print(outputs.last_hidden_state)
'''

# 문장 하나만 BERT에 넣어 테스트해보기

# 1. features_array[]에 들어있는 첫 번째 문장(세 개의 feature를 가지는 tensor)을 프린트해 확인
print(features_array[0])

# 2. 위에서 세팅해둔 BERT모델에 첫 번째 문장 넣기
outputs = model(**features_array[0])

# 3. outputs.keys()를 통해 output에 어떤 키들이 있나 확인
print(outputs.keys())

# 4. last_hidden_state : 마지막 은닉층
print(outputs.last_hidden_state)
print(outputs.last_hidden_state.shape) # last_hidden_state의 shape는 torch.Size([1,7,768]) 이다

# 5. pooler_output : 마지막 은닉층의 첫 번째 토큰인 [CLS]토큰의 embedding 이다.
print(outputs.pooler_output)
print(outputs.pooler_output.shape) # pooler_output의 shape는 torch.Size([1,768]) 이다