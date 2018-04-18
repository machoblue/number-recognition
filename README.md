# number-recognition

## 利用手順

### VMインスタンスの作成
以下のVMインスタンスを作成します。

|項目|設定値|
|-|-|
|OS|Debian9|
|ゾーン|asia-east1-a|
|ファイアウォール|HTTPトラフィックを許可するにチェック|

### 環境の整備

まずrootユーザーになります。

```
$ sudo -i
```

次に必要なパッケージをインストールします。

```
\# apt-get update
\# apt-get install -y python3-pip git
```

GitHubからアプリを取得します。そして、必要なpythonのパッケージをインストールします。

```
\# git clone https://github.com/machoblue/number-recognition
\# cd number-recognition
\# pip3 install -r requirements.txt
```

### 学習

以下を実行し、学習します。

```
\# python3 ch05/train_neuralnet.py
```

### 推論

以下を実行し、アプリを起動します。

```
\# cp -a webapps /opt/
\# mkdir /uploads

\# cp numberrec.service /etc/systemd/system/

\# systemctl daomon-reload
\# systemctl enable numberrec
\# systemctl start numberrec
\# systemctl status numberrec
```

以上です。