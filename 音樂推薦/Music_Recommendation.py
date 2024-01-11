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

            predicted_genre = predict_genre(filepath)
            recommended_links = get_youtube_links(predicted_genre)
            # 获取Spotify推荐链接
            try:
                spotify_results = sp.recommendations(
                    seed_genres=[predicted_genre], limit=5)
                spotify_links = [track['external_urls']['spotify']
                                 for track in spotify_results['tracks']]
            except Exception as e:
                print(f"Error fetching Spotify recommendations: {e}")
                spotify_links = []  # 如果出现错误，设置为空列表

            # 向模板传递YouTube和Spotify链接
            return render_template('result.html', genre=predicted_genre, youtube_links=recommended_links, spotify_links=spotify_links)
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


model = build_model()  # 创建模型实例
model.load_weights('100_epoch_tr_GRU6.cpkt')  # 加载权重


# def preprocess_audio_to_image(filepath):
#     # 加载音频文件
#     x, sr = librosa.load(filepath)
#     # 提取音频特征（例如 STFT）
#     X = librosa.stft(x)
#     Xdb = librosa.amplitude_to_db(abs(X))
#     # 调整特征的尺寸以匹配模型输入
#     resized_Xdb = cv2.resize(Xdb, (256, 256))
#     # 转换为3通道图像（如果模型需要三通道输入）
#     image_input = np.repeat(resized_Xdb[..., np.newaxis], 3, axis=-1)
#     return image_input

# def predict_genre(filepath):
#     # 将音频文件预处理为图像格式
#     image_input = preprocess_audio_to_image(filepath)
#     # 添加一个批次维度，因为模型预测需要批量数据
#     input_data = np.expand_dims(image_input, axis=0)
#     # 使用模型进行预测
#     predictions = model.predict(input_data)
#     # 解码预测结果
#     genre = decode_predictions(predictions)
#     return genre


def preprocess_audio_to_image(filepath):
    # 加载音频文件
    x, sr = librosa.load(filepath)

    # 使用第二段代码中的方法生成频谱图
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    Xdb = cv2.resize(Xdb, (256, 256))  # 确保尺寸与模型输入匹配
    Xdb_normalized = (Xdb - np.min(Xdb)) / (np.max(Xdb) - np.min(Xdb))
    image_input = cv2.merge([Xdb_normalized, Xdb_normalized, Xdb_normalized])

    return image_input


def predict_genre(filepath):
    # 将音频文件预处理为图像格式
    image_input = preprocess_audio_to_image(filepath)
    # 添加一个批次维度，因为模型预测需要批量数据
    input_data = np.expand_dims(image_input, axis=0)
    # 使用模型进行预测
    predictions = model.predict(input_data)
    # 解码预测结果
    genre = decode_predictions(predictions)
    return genre


def decode_predictions(predictions):
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']  # 音乐类别列表
    predicted_index = np.argmax(predictions)
    return genres[predicted_index]


# 根據音樂類型返回推薦的YouTube連結


def get_youtube_links(genre, max_results=5):
    # You need to get this from the Google Cloud Console.
    API_KEY = 'AIzaSyCogkMHWVJQbp2pdzSCwzQa4rgSFMOUElI'
    YOUTUBE_SEARCH_URL = 'https://www.googleapis.com/youtube/v3/search'

    params = {
        'part': 'snippet',
        'q': genre + ' music',
        'type': 'video',
        'key': API_KEY,
        'maxResults': max_results
    }

    response = requests.get(YOUTUBE_SEARCH_URL, params=params)
    data = response.json()
    video_links = ['https://www.youtube.com/watch?v=' +
                   item['id']['videoId'] for item in data['items']]

    return video_links


if __name__ == "__main__":
    app.run(debug=True)
