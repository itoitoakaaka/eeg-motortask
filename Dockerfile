# Python 3.11をベースにする
FROM python:3.11-slim

# OSの最低限必要なツールを入れる
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# コンテナ内の作業場所を決める
WORKDIR /app

# ライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 手元のファイルをコンテナにコピー
COPY . .