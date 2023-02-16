import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import re
# Pillow
from PIL import Image

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

stop_words = set(stopwords.words("english"))

def extract_nouns(text):
    words = word_tokenize(text)
    tagged = nltk.pos_tag(words)
    nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return " ".join(nouns)

def extract_words(text):
    words = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(words)
    target_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    target_words = [word for word, pos in tagged if pos in target_pos]
    return " ".join(target_words)


def calculate_tfidf(list_all, text_1):
    # TfidfVectorizerのインスタンスを生成
    vectorizer = TfidfVectorizer(stop_words="english")
    # list_allを学習
    X = vectorizer.fit_transform(list_all)
    # text_1にTfidfモデルを適用
    response = vectorizer.transform([text_1])
    # Tfidfの結果を確認
    features = vectorizer.get_feature_names_out()
    scores = response.toarray()[0]
    result = dict(zip(features, scores))
    return result

def color_func(word, font_size, position, orientation, random_state, font_path):
    return 'black'

def plot_wordcloud_black(result, title):
    # wordcloudを描画
    wordcloud = WordCloud(
        color_func=color_func, background_color='white', width=600, height=400, max_words=100).generate_from_frequencies(result)
    fig = plt.figure(figsize=(8, 6), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    #plt.title(f"Title: {title}")
    plt.close()
    fig.savefig("wordcloud.png")
    img = Image.open('wordcloud.png')
    st.image(img, caption='WordCloud: '+title, use_column_width=True)
    
@st.cache_data
def load_data():
    filename = "df_medium_2023021300.csv"
    df = pd.read_csv(filename)
    return df

def main():
    df = load_data()
    #st.header("Word Cloud")
    st.title('Medium article summarizer')

    # 使用するテキストのリスト
    texts = df['texts']

    # サイドバー（入力画面）
    st.sidebar.header('Input Features')

    # Tfidfを適用するテキストの番号を選択する
    n = st.sidebar.slider("WordCloudを描くデータ番号を選択して下さい。",
                  0, len(texts) - 1, 0)

    # Tfidfを適用するテキスト
    text_1 = texts[n]

    # タイトルを表示する。
    title = df['titles'][n]

    # Tfidfを適用するテキスト以外のテキストのリスト
    #list_all = set(texts[0:len(texts)]) - set(texts[n])
    #list_all = list(list_all)
    list_all = texts

    # Tfidfを計算する
    tfidf_result = calculate_tfidf(list_all, text_1)

    # メインパネル
    st.write('### Title: ', title)

    # Wordcloudを描画する
    plot_wordcloud_black(tfidf_result, title)

if __name__ == '__main__':
    main()
