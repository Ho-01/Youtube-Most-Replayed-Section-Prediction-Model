from youtube_transcript_api import YouTubeTranscriptApi

'''
import nltk
nltk.download()
''' # NLTK 다운로드는 첫 1회만 수행하면 된다

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# URL : https://www.youtube.com/watch?v=X7158uQk1yI
# Title : FULL MATCH: Brazil vs. France 2006 FIFA World Cup
transcript_array = YouTubeTranscriptApi.get_transcript("X7158uQk1yI")
print(transcript_array)


# transcript 에 ["문장1", "문장2", "문장3", .... ] 의 형태로 저장
transcript = []
for i in range(0, len(transcript_array)):
    print(transcript_array[i]['text'])
    transcript.append(transcript_array[i]['text'])


# word_tokenized_transcript 에 [ ["단어1", "단어2", ...], ["단어3", "단어4", ...], [...], .... ] 의 형태로 토큰화해 저장

# 토큰화하면서 불용어도 같이 제거하기
stop_words = set(stopwords.words('english'))

word_tokenized_transcript = []
for i in range(0, len(transcript)):
    # 먼저 각 문장을 단어로 토큰화하고
    tmp = word_tokenize(transcript[i])

    # 불용어를 제거
    result = []
    for word in tmp:
        if word not in stop_words:
            result.append(word)

    # 토큰화+불용어 제거 완료된 문장을 word_tokenized_transcript 에 append
    word_tokenized_transcript.append(result)

print(word_tokenized_transcript)



