<h1 align="center">
  LADA REALTIME PLAYER
</h1>

***
## 機能
 
動画をお手軽にLADA処理させながら再生します。     

***
## 前提環境
 
 下記ソフトウェアをインストールしたWindows 11
 
* Lada Portable for Windows packed 250915
* [VLC media player 64bit版](https://www.videolan.org/vlc/index.ja.html)  

***
## インストール方法
 
* `rp_pf.py`　をLadaインストールフォルダにコピー 
* 追加パッケージ4つをインストール  
 Ladaインストールフォルダで、  
  
　`.\python\python.exe -m pip install PyQt6 PyOpenGL PyOpenGL-accelerate python-vlc`  
  
  
以下のファイルを<strong>バックアップ</strong>した後、ここで提供したものを上書きする。  

　`\python\Lib\site-packages\lada\lib\frame_restorer.py`  
  
  ※オリジナル処理はそのままですので`lada-cli.exe`等への影響は皆無です。  

***
## 起動方法 

　Ladaインストールフォルダから   
  
　`.\python\python.exe rp_pf.py`
  
  ※VLC関連のエラーが出る方は、一旦、`python-vlc`をuninstallしてから起動してください。  
  音がでなくなりますが動作します。  

***
## 操作方法

* 停止再開：画面クリック、SPACE
* 再生位置移動：進捗バーをクリック、0-9,S,E、矢印キー、h,j,k,l,;
* 範囲指定：開始登録　Ctrl+S、終了登録　Ctrl+E、再生モードトグル Ctrl+P 
* フルスクリーン：ダブルクリック、F
* ミュートトグル：M
* AI処理ON/OFF：X
* D&D対応
 
設定で多少チューニングができます。  
  
<おすすめ設定値>   
  
<strong><並列クリップ処理></strong> 8  大きすぎるとモザイク処理漏れ確率があがります。 
<strong><バッチサイズ></strong> 8    
<strong><最大クリップ長></strong> 8    
  
  
キャッシュ管理を行っていますが効果は薄いです。  


***
## 制限事項 

LADA処理を並列化しましたがバグありのためモザイク処理漏れが頻発します。  （先行フレーム処理が後方フレーム処理より遅れた場合の扱いによるバグ）  
音は出ますが調整要です。     

***
## PR歓迎します 

今回、LADA処理自体を一部変更して高速化を実現しました。  
すべてのソースが公開されていますので技術力がある方でしたらさらなる高速化が可能です。  
協力しながら良いものに仕上げていけたらと思っています。  

***
## ライセンス

  LADAが[AGPLv3](https://ja.wikipedia.org/wiki/GNU_Affero_General_Public_License)ライセンスなのでコピーレフトが適用されこのソフトウェアもAGPLv3となります。  

*** 
## 検証PC

Windows 11 、CPU Ryzen 9 3900X/Ryzen 7 3700X/Core-i5 9600K、GPU RTX 2060 6GB/3060 12GB/4070 12GB で動作確認しました。 

*** 
## 謝辞

This project builds upon work done by these fantastic individuals and projects:

* [lada](https://github.com/ladaapp): Restore videos with pixelated/mosaic regions.
* [DeepMosaics](https://github.com/HypoX64/DeepMosaics): Provided code for mosaic dataset creation. Also inspired me to start this project.
* [BasicVSR++](https://ckkelvinchan.github.io/projects/BasicVSR++) / [MMagic](https://github.com/open-mmlab/mmagic): Used as the base model for mosaic removal.
* [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): Used their image degradation model design for our mosaic detection model degradation pipeline.
* PyTorch, FFmpeg, GStreamer, GTK and [all other folks building our ecosystem](https://xkcd.com/2347/)
