from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BertTokenizer
from transformers import BertConfig, BertModel
import torch

# BERT 모델 초기설정 : "bert-base-uncased" 로 버전 통일
pretrained_model_config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", config=pretrained_model_config)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Bert-base의 토크나이저



# URL을 넣어주면 youtube_transcript_api를 이용해 transcript를 받아오는 함수
# 주의할점 : "https://www.youtube.com/watch?v=X7158uQk1yI" 같이 "v="를 통해 비디오id가 구분되는 youtube URL이어야 함
def get_transcript_from_url(url):
    video_id = url.split("v=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    print(transcript) # 출력테스트 transcript : [{'text': "자막1...", 'start': 0.0, 'duration': 4.279}, {'text': "자막2...", 'start': 2.34, 'duration': 4.979}, { ... } , { ... } , ... ]
    return transcript



# transcript[]를 넣어주면, timestamp와 duration을 제거하고 text만 들어있는 sentence_array[]를 반환
def get_caption_array_from_transcript(transcript):
    caption_array = []
    for element in transcript:
        caption_array.append(element["text"])

    print(caption_array) # 출력테스트 caption_array : [ "자막1..", "자막2...", "자막3...", ... "자막n..." ]
    return caption_array



# caption_array의 각 caption은 'input_ids', 'token_type_ids', 'attention_mask' 세 개의 tensor 피처를 가지는 딕셔너리로 바뀌어 features_array에 저장된다.
# 이 과정은 BERT토크나이저에 의해 수행되며, 같은 버전의 BERT모델에 넣어서 각 문장을 embedding하기 위한 데이터 전처리 과정이다.
def get_features_array_from_caption_array(caption_array):
    features_array = []
    for sentence in caption_array:
        features_array.append(tokenizer(sentence, max_length=17, padding="max_length", truncation=True,
                                        return_tensors="pt"))  # return_tensors를 "pt"로 주어 피처를 토치 텐서로 변환

    print(features_array) # 출력테스트 features_array
    return features_array



# 주석주석
def embedding_X(url):
    transcript = get_transcript_from_url(url)
    caption_array = get_caption_array_from_transcript(transcript)
    features_array = get_features_array_from_caption_array(caption_array)

    embedded_captions = []
    # BERT에 태우기
    for feature in features_array:
        outputs = model(**feature)
        print(outputs.pooler_output)
        embedded_captions.append(outputs.pooler_output)
    return embedded_captions



# 테스트용 URL
# Title : FULL MATCH: Brazil vs. France 200  v6 FIFA World Cup
test_URL = "https://www.youtube.com/watch?v=X7158uQk1yI"


# 테스트 코드 : 주석풀고 실행
embedding_X(test_URL)


# 문장 하나만 BERT에 넣어 테스트해보기 : 주석풀고 실행
'''

transcript = get_transcript_from_url(test_URL)
caption_array = get_caption_array_from_transcript(transcript)
features_array = get_features_array_from_caption_array(caption_array)

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

'''