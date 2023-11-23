from word_tokenize import word_tokenized_transcript

from gensim.models import Word2Vec


num_workers = 4 # 실행할 병렬 프로세스의 수, 코어수. 주로 4~6 사이 지정
min_word_count = 5 # 단어에 대한 최소 빈도수. min_count=5라면 빈도수 5 이하 무시
size = 300 # 각 단어에 대한 임베딩 된 벡터 차원 정의, size=2라면 한 문장의 벡터는 [-0.1248574, 0.255778]과 같은 형태를 가지게 된다.
window = 10 # 문맥 윈도우 수. 양쪽으로 몇 개의 단어까지 고려해서 의미를 파악할 것인지 지정하는 것
sample = 1e-3 # 빠른 학습을 위해 정답 단어 라벨에 대한 다운샘플링 비율을 지정하는 것, 보통 0.001이 좋은 성능을 냄

# 단어_토큰화된 문장들을 이용해서 벡터를 생성한다
# Word2Vec( sentences, workers, size, min_count, window, sample)
model = Word2Vec(sentences=word_tokenized_transcript,
                 workers=num_workers,
                 vector_size=size,
                 min_count=min_word_count,
                 window=window,
                 sample=sample)


model_result = model.wv.most_similar("Brazil")
print(model_result)

