from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from datetime import datetime, timedelta
import random
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, GRU, Flatten, Bidirectional, Input
import pickle
import os
from flask import Flask, render_template, request, redirect, url_for, flash
import librosa
from keras.models import load_model
import keras
import numpy as np
from sklearn.metrics import classification_report
import librosa.display
import requests
import cv2
import matplotlib.pyplot as plt
import os

# 配置Flask應用程序
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 檢查文件是否是允许的音樂格式


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 初始化Spotify客户端
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="8ca65d108d0b4ca7a338a10c35ef06f6",
                                                           client_secret="246501f224f548fdbfc4da153617bb8a"))

# 主要的音樂上傳和處理路由


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # 預測音樂流派
            predicted_genre = predict_genre(filepath)
            # 獲取用戶選擇地區以及時間範圍
            region = request.form['region']
            time_range = request.form['time_range']
            # 根據流派和用戶選擇的地區來給 YouTube 連結
            recommended_links = get_youtube_links(
                predicted_genre, region, time_range)
            # 獲取Spotify連結
            try:
                spotify_results = sp.recommendations(
                    seed_genres=[predicted_genre], limit=5)
                spotify_links = [track['external_urls']['spotify']
                                 for track in spotify_results['tracks']]
            except Exception as e:
                print(f"Error fetching Spotify recommendations: {e}")
                spotify_links = []  # 如果出现错误，设置为空列表

            # 向模板传递YouTube和Spotify链接
            return render_template('result.html', genre=predicted_genre, youtube_links=recommended_links, spotify_links=spotify_links, region=region, time_range=time_range)
    return render_template('upload.html')

# 2. 加载預先訓練的模型


def build_model():
    mobilenetv2 = MobileNetV2(input_shape=(
        256, 256, 3), include_top=False, weights="imagenet")
    mobilenetv2.trainable = False

    input_image = Input(shape=(256, 256, 3))
    encoded_image = mobilenetv2(input_image)
    global_avg_pooling = GlobalAveragePooling2D()(encoded_image)
    gru_input = tf.keras.layers.Reshape((1, -1))(global_avg_pooling)
    gru_layer = Bidirectional(
        GRU(512, return_sequences=True, dropout=0.7, recurrent_dropout=0.7))(gru_input)
    gru_output = Flatten()(gru_layer)
    output = Dense(10, activation='softmax')(gru_output)

    model = Model(inputs=input_image, outputs=output)
    return model


model = build_model()  # 創建模型
model.load_weights('100_epoch_tr_GRU6.cpkt')  # 加載權重


def preprocess_audio_to_image(filepath):
    # 加载音频文件
    x, sr = librosa.load(filepath)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb)
    temp_image_path = 'temp_image.png'
    plt.savefig(temp_image_path)
    plt.close()
   # 使用cv2.imread读取保存的图像文件，并转换为RGB格式
    img_arr = cv2.imread(temp_image_path)[..., ::-1]
    # 调整图像大小
    resized_arr = cv2.resize(img_arr, (256, 256))
    return resized_arr


def predict_genre(filepath):
    # 將音頻作預處理成圖像
    image_input = preprocess_audio_to_image(filepath)
    # 添加一個維度
    input_data = np.expand_dims(image_input, axis=0)
    # 使用預測
    predictions = model.predict(input_data)
    # 預測结果
    genre = decode_predictions(predictions)
    return genre


def decode_predictions(predictions):
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    predicted_index = np.argmax(predictions, axis=1)[0]
    return genres[predicted_index]


# 根據音樂類型返回推薦的YouTube連結

def get_youtube_links(genre, region, time_range,  max_results=5):
    # You need to get this from the Google Cloud Console.
    API_KEY = 'AIzaSyCogkMHWVJQbp2pdzSCwzQa4rgSFMOUElI'
    YOUTUBE_SEARCH_URL = 'https://www.googleapis.com/youtube/v3/search'
    if region == 'English':
        search_query = 'English' + genre + ' music' + time_range
    elif region == 'Japanese':
        search_query = 'Japanese' + genre + ' 音楽' + time_range
    elif region == 'Korean':
        search_query = 'Korean' + genre + ' 음악' + time_range
    elif region == 'Chinese':
        search_query = 'Chinese' + genre + ' 音樂' + time_range
    elif region == 'Russian':
        search_query = 'Russian' + genre + ' музыка' + time_range
    else:
        search_query = genre + ' music'  # 默认为英文音乐
    params = {
        'part': 'snippet',
        'q': search_query,  # 使用search_query变量
        'type': 'video',
        'key': API_KEY,
        'maxResults': max_results
    }

    response = requests.get(YOUTUBE_SEARCH_URL, params=params)
    data = response.json()

    # 创建空列表来存储链接和预览图像的元组
    video_links_with_preview = []

    # 遍历搜索结果
    for item in data['items']:
        video_id = item['id']['videoId']
        video_link = f'https://www.youtube.com/watch?v={video_id}'

        # 获取预览图像的 URL
        preview_image_url = item['snippet']['thumbnails']['default']['url']

        # 将视频链接和预览图像 URL 组成元组，添加到列表中
        video_links_with_preview.append((video_link, preview_image_url))

    return video_links_with_preview


if __name__ == "__main__":
    app.run(debug=True)
