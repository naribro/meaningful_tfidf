# 본 Paper 관련된 모든 패키지는 한 번에 로딩하기
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

import gensim
import re
import operator

# 최초 코드 실행시 다운로드는 한 번만 실행하면, 다음부터는 실행할 필요 없음
nltk.download('stopwords')
nltk.download('wordnet')


###################################
#
#  1. Dataset 불러오기
#
###################################


## BBC News Summary 불러오기 (5개 카테고리)
news_categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
news_cnt = {'business':510, 'entertainment': 386, 'politics':417, 'sport': 511, 'tech':401}
news_files = {'business':[], 'entertainment':[], 'politics':[], 'sport':[], 'tech':[]}

file_path_root = '1. Dataset/D1. BBC News Summary/'

for category in news_categories:
    print(category)
    for idx in range(1, news_cnt[category] + 1):
        path = file_path_root + category + '/' + str(idx).zfill(3) + '.txt'  # zfill : 텍스트의 앞을 0으로 채워줌
        with open(path, 'r', encoding="utf-8") as file:
            lines = file.readlines()
        news_files[category].append(lines)


###################################
#
#  2. 텍스트 전처리(tokenizer, stemming, tokenizer 등)
#
###################################


# 파일에서 불러온 라인을 토크나이징하여 DataFrame에 넣기
news = pd.DataFrame(columns=['category', 'file_idx', 'sen_idx', 'sen'] )

for category in news_categories:
    print(category, "start ")
    for file_idx, file in enumerate(news_files[category]):
        sen_idx = 0
        for line in file:
            if line != '\n':
                sentences = sent_tokenize(line)    # 문장이 여러개일 경우, 문장별로 나눠주기
                for sen in sentences:
                    news = news.append({ 'category':category, 'file_idx': file_idx, 'sen_idx': sen_idx, 'sen': sen}, ignore_index=True)
                    sen_idx = sen_idx + 1
    print(category, "end")


# 영문 텍스트 전처리 함수 : 영문자 공백처리, 소문자로 변환, Stopword 불용어 제거, Stemmer, Lemmatizer 적용
# Stopwords는 TFIDF Vertorizer에서 수행해주므로 여기서는 불필요

# stops = set(stopwords.words('english'))                              # stopword를 세트로 변환
stemmer = nltk.stem.PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


def preprocessing(sentence):
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)                     # 영문자가 아닌 문자는 공백 처리
    sentence = sentence.lower().split()                               # 소문자 변환
#     sentence = [w for w in sentence if not w in stops]               # Stopword 불용어 제거
#     sentence = [stemmer.stem(w) for w in sentence]                    # Stemmer 적용(어간 추출)
#     sentence = [wordnet_lemmatizer.lemmatize(w) for w in sentence]    # Lemmatizatizer 적용(동음이의어 처리)
    sentence = ' '.join(sentence)                                     # 공백으로 구분된 문자열로 결합하여 결과를 반환
    return sentence


# DataFrame에 넣은 문장에 전처리 적용하여 sen2 컬럼에 넣기
news['sen2'] = news.sen.apply(preprocessing)



###################################
#
#  3. TF-IDF 학습하기
#
###################################

# Tokenizing된 컬럼(sen2)으로 TF-IDF 만들기
tfidfVectorizer= TfidfVectorizer(tokenizer=None, min_df=0.001, use_idf=True, stop_words='english')  # Best
news_tfidf = tfidfVectorizer.fit_transform(news['sen2'])


###################################
#
#  4. Word2Vec Load / 유사도 Matrix 만들기
#
###################################

print(len(tfidfVectorizer.vocabulary_))

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./2. Embedding/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
print(word2vec_model.vectors.shape)


# TF-IDF vectorizer가 생성한 vocabulary 기준으로 Word2vec 유사도 매트릭스를 생성
def get_word_sim_arr(vocab=tfidfVectorizer.vocabulary_):
    vocab_size = len(vocab)
    sim_arr = np.zeros((vocab_size, vocab_size))  # 단어간 유사도 값을 보관할 배열 생성

    for key_i in vocab.keys():
        if not key_i in word2vec_model:
            continue

        for key_j in vocab.keys():
            if not key_j in word2vec_model:
                continue

            i_idx = vocab[key_i]
            j_idx = vocab[key_j]
            sim_arr[i_idx, j_idx] = word2vec_model.wv.similarity(key_i, key_j)

    return sim_arr


word_sim_arr = get_word_sim_arr(tfidfVectorizer.vocabulary_)


###################################
#
#  5. 문장 - 문장간 유사도 매트릭스 생성 및 측정
#
###################################

# dense_output=False로 CSR Matrix 생성(메모리 절약)
news_sim_arr = cosine_similarity(news_tfidf, news_tfidf, dense_output=False)


# 문장 - 문장간 유사도 매트릭스 생성 (word_sim_arr 입력)
def get_sen_sim_w2v_arr(word_sim_arr, sim_thres=0.5, score_thres=0.5, zero_flag='self'):
    print("# get_sen_sim_w2v_arr")
    print("# sim_thres = ", sim_thres, " score_thres = ", score_thres, " zero_flag = ", zero_flag)
    word_sim_arr_adj = np.where(word_sim_arr < sim_thres, 0, word_sim_arr)  # np.where(조건, 참, 거짓)
    news_tfidf_w2v = news_tfidf.copy().tolil()

    # news_tfidf_w2v을 한 행씩 추출
    for v_idx, v in enumerate(news_tfidf_w2v):
        rows = v.rows  # CSR Matrix : v.indices, v.data, LIL Matrix : v.rows[0], v.data[0] 으로 사용
        data = v.data  # rows는 index, data는 score

        for idx, score in zip(rows[0], data[0]):
            sim_vec = word_sim_arr_adj[idx].copy() * score
            sim_vec = sim_vec.reshape(1, sim_vec.shape[0])  # news_tfidf_wv 에 더해주기 위해 2차원 배열로 변경

            # idx 자신의 유사도를 0으로 만들거나, 문장 내 전체 단어의 유사도를 0으로 만듬
            if zero_flag == 'self':
                sim_vec[0, idx] = 0
            elif zero_flag == 'all':
                sim_vec[0][rows[0]] = 0

            # sim_vec의 score가 thres 미만이면 0으로 버림
            sim_vec = np.where(sim_vec < score_thres, 0, sim_vec)
            news_tfidf_w2v[v_idx] += sim_vec

    return news_tfidf_w2v


# 문장-문장 단위 비교
def get_sen_sim_score(news_sim_arr, sen_sim_thres=0.3, cat_thres=0.4, cat_thres_method='rank'):
    print("# get_sen_sim_score")
    print("sen_sim_thres = ", sen_sim_thres, " cat_thres = ", cat_thres, " cat_thres_method = ", cat_thres_method)
    dt_cnt_cat = {'business': 0, 'entertainment': 0, 'politics': 0, 'sport': 0, 'tech': 0, 'unknown': 0}
    dt_cnt_tf = {'True': 0, 'False': 0, 'unknown': 0}

    label = []
    predict = []

    for i in range(len(news)):
        best = news_sim_arr[i][0].toarray()[0]
        best[i] = 0

        if any(best > sen_sim_thres):  # 문장간 유사도에 대한 threshold
            cnt = news.iloc[best > sen_sim_thres]['category'].value_counts()
            cat = cnt.keys()[0]
            score = cnt.values[0]
            score_sum = cnt.values.sum()
            score_ratio = score / score_sum

            if cat_thres_method == 'ratio':
                if score_ratio < cat_thres:  # 가장 많이 나온 카테고리의 점유율 thres 적용
                    cat = "unknown"
            elif cat_thres_method == 'rank':
                if len(cnt.keys()) > 2 and (cnt.keys()[0] == cnt.keys()[1]):
                    cat = 'unknown'

            dt_cnt_cat[cat] = dt_cnt_cat[cat] + 1

            prov_flag = ''
            if cat != 'unknown':
                label.append(news['category'][i])
                predict.append(cat)

                if news['category'][i] == cat:
                    prov_flag = 'True'
                else:
                    prov_flag = 'False'
            else:
                prov_flag = 'unknown'
            dt_cnt_tf[prov_flag] = dt_cnt_tf[prov_flag] + 1


        else:
            cat = 'unknown'
            dt_cnt_cat["unknown"] = dt_cnt_cat["unknown"] + 1
            dt_cnt_tf['unknown'] = dt_cnt_tf['unknown'] + 1

    print(dt_cnt_cat)
    print(dt_cnt_tf)

    return label, predict


# 기본 TF-IDF 모델로 문장-문장 유사도 매트릭스 평가
label_sen_origin, predict_sen_origin = get_sen_sim_score(news_sim_arr)

# 성능평가
print(" 1. 기본 TF-IDF 모델로 문장-문장 유사도 매트릭스 평가")
print("   - Acc      : ", accuracy_score(label_sen_origin, predict_sen_origin))	# 0.3
print("   - Recall   : ", recall_score(label_sen_origin, predict_sen_origin, average='weighted'))	# 0.42
print("   - Precison : ", precision_score(label_sen_origin, predict_sen_origin, average='weighted'))	# 0.5
print("   - F1-Score : ", f1_score(label_sen_origin, predict_sen_origin, average='weighted'))	# 0.46

# W2V이 결합된 TF-IDF 모델로 매트릭스 평가 - Best 모델
news_tfidf_w2v = get_sen_sim_w2v_arr(word_sim_arr)  # 기본 : sim_thres=0.5, score_thres=0.5
news_sim_w2v_arr = cosine_similarity(news_tfidf_w2v, news_tfidf_w2v, dense_output=False)
label_sen_w2v, predict_sen_w2v = get_sen_sim_score(news_sim_w2v_arr)

# 성능평가
print(" 2. W2V이 결합된 TF-IDF 모델로 문장-문장 유사도 매트릭스 평가")
print("   - Acc      : ", accuracy_score(label_sen_w2v, predict_sen_w2v))	# 0.3
print("   - Recall   : ", recall_score(label_sen_w2v, predict_sen_w2v, average='weighted'))	# 0.42
print("   - Precison : ", precision_score(label_sen_w2v, predict_sen_w2v, average='weighted'))	# 0.5
print("   - F1-Score : ", f1_score(label_sen_w2v, predict_sen_w2v, average='weighted'))	# 0.46


###################################
#
#  6. 문서 - 문장간 유사도 매트릭스 생성 및 측정
#
###################################

# 문서의 범위 만들기
news_file_range = []
start = 0
cur_file_idx = news['file_idx'][0]
for idx, r in news.iterrows():
    if r['file_idx'] != cur_file_idx:
        news_file_range.append((start, idx-1))
        start = idx
        cur_file_idx = r['file_idx']


# news_sim_mx를 문서 단위로 더해주기
def get_doc_sim_arr(sim, mean_method='nan'):
    print("# get_doc_sim_arr")
    print("# mean_method = ", mean_method)

    thres = 0.5
    doc_arr = np.zeros((sim.shape[0], sum(news_cnt.values())))

    for idx, ran in enumerate(news_file_range):
        row = sim[:, ran[0]:ran[1] + 1].toarray()

        if mean_method == 'all':
            row = row.mean(axis=1)
        elif mean_method == 'nan':
            row = np.where(row < thres, np.nan, row)
            row = np.nanmean(row, axis=1)

        doc_arr[:, idx] = row

    return doc_arr


# 문서-문장 단위 비교
def get_doc_sim_score(news, sim_arr, token_len=0, sen_sim_thres=0.5, cat_thres=0.4, cat_thres_method='ratio'):
    print("# get_doc_sim_score")
    print("token_len = ", token_len, " sen_sim_thres = ", sen_sim_thres, " cat_thres = ", cat_thres
          , " cat_thres_method = ", cat_thres_method)

    dt_cnt_cat = {'business': 0, 'entertainment': 0, 'politics': 0, 'sport': 0, 'tech': 0, 'unknown': 0}
    dt_cnt_tf = {'True': 0, 'False': 0, 'unknown': 0}

    label = []
    predict = []

    for c in news_categories:
        news['cnt_' + c] = 0

    for i in range(len(news)):
        best = sim_arr[i]

        if len(news.iloc[i]['sen2'].split()) >= token_len and any(best > sen_sim_thres):  # len 조정
            cnt = {}
            r = np.where(best > sen_sim_thres)[0].tolist()
            for ran in np.array(news_file_range)[r]:
                category = news['category'][ran[0]]
                if category not in cnt.keys():
                    cnt[category] = 1
                else:
                    cnt[category] += 1

            # 가장 순위가 높은 category 추출, 점유율 계산
            cnt = dict(sorted(cnt.items(), key=operator.itemgetter(1), reverse=True))
            cat = list(cnt.keys())[0]
            score = list(cnt.values())[0]
            score_sum = sum(cnt.values())
            score_ratio = score / score_sum

            for k, v in cnt.items():
                news.loc[i, 'cnt_' + k] = v

            # unknown 처리
            if cat_thres_method == 'ratio':
                if score_ratio < cat_thres:
                    cat = "unknown"
            elif cat_thres_method == 'rank':
                if len(cnt.keys()) > 2 and (cnt.keys()[0] == cnt.keys()[1]):
                    cat = 'unknown'

            dt_cnt_cat[cat] = dt_cnt_cat[cat] + 1

            # 정답 여부를 check
            prov_flag = ''
            if cat != 'unknown':
                label.append(news['category'][i])
                predict.append(cat)
                if news['category'][i] == cat:
                    prov_flag = 'True'
                else:
                    prov_flag = 'False'
            else:
                prov_flag = 'unknown'
            dt_cnt_tf[prov_flag] = dt_cnt_tf[prov_flag] + 1


        else:
            dt_cnt_cat["unknown"] = dt_cnt_cat["unknown"] + 1
            dt_cnt_tf['unknown'] = dt_cnt_tf['unknown'] + 1

    print(dt_cnt_cat)
    print(dt_cnt_tf)

    return news, label, predict


# TF-IDF 기반 문장 - 문서간 유사도 매트릭스 구하기
news_doc_sim_arr = get_doc_sim_arr(news_sim_arr)
news, label_doc_origin, predict_doc_origin = get_doc_sim_score(news, news_doc_sim_arr)

# 성능 측정
print(" 3. 기본 TF-IDF 모델로 문장-문서간 유사도 매트릭스 평가")
print("   - Acc      : ", accuracy_score(label_doc_origin, predict_doc_origin))	# 0.3
print("   - Recall   : ", recall_score(label_doc_origin, predict_doc_origin, average='weighted'))	# 0.42
print("   - Precison : ", precision_score(label_doc_origin, predict_doc_origin, average='weighted'))	# 0.5
print("   - F1-Score : ", f1_score(label_doc_origin, predict_doc_origin, average='weighted'))	# 0.46

# W2V이 결합된 문장 - 문서간 유사도 매트릭스 구하기
# get_sim_wv_mx(sim_thres=0.5, score_thres=0.5, zero_flag=True) 기준
news_doc_sim_w2v_arr = get_doc_sim_arr(news_sim_w2v_arr)
news, label_doc_w2v, predict_doc_w2v = get_doc_sim_score(news, news_doc_sim_w2v_arr)

# 성능 측정
print(" 4. W2V이 결합된 TF-IDF 모델로 문장-문서간 유사도 매트릭스 평가")
print("   - Acc      : ", accuracy_score(label_doc_w2v, predict_doc_w2v))	# 0.3
print("   - Recall   : ", recall_score(label_doc_w2v, predict_doc_w2v, average='weighted'))	# 0.42
print("   - Precison : ", precision_score(label_doc_w2v, predict_doc_w2v, average='weighted'))	# 0.5
print("   - F1-Score : ", f1_score(label_doc_w2v, predict_doc_w2v, average='weighted'))	# 0.46

