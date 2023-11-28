from word_tokenize import word_tokenized_transcript

from gensim.models import Word2Vec


workers = 4 # 실행할 병렬 프로세스의 수, 코어수. 주로 4~6 사이 지정
min_count = 1 # 단어에 대한 최소 빈도수. min_count=5라면 빈도수 5 이하 무시
vector_size = 300 # 각 단어에 대한 임베딩 된 벡터 차원 정의, size=2라면 한 문장의 벡터는 [-0.1248574, 0.255778]과 같은 형태를 가지게 된다.
window = 20 # 문맥 윈도우 수. 양쪽으로 몇 개의 단어까지 고려해서 의미를 파악할 것인지 지정하는 것
sample = 1e-3 # 빠른 학습을 위해 정답 단어 라벨에 대한 다운샘플링 비율을 지정하는 것, 보통 0.001이 좋은 성능을 냄
sg = 0 # 0은 CBOW방식, 1은 skip-grows 방식

# 단어_토큰화된 문장들을 이용해서 벡터를 생성한다
# Word2Vec( sentences, workers, size, min_count, window, sample)
model = Word2Vec(sentences=word_tokenized_transcript,
                 workers=workers,
                 min_count=min_count,
                 vector_size=vector_size,
                 window=window,
                 sample=sample,
                 sg=sg)

print(model.cum_table)
model_result = model.wv.most_similar("final")
print(model_result)
# print(model.wv['score'])


