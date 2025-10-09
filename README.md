<h1 align="center">
  LADA REALTIME PLAYER
</h1>

***
## 機能
 
お手元の動画をお手軽にLADA処理させながら再生します！   
頻繁に止まったり、カクついたります。音ズレも結構あります。   
それでも楽しめると思います。   
それでも良いと思ってくださる寛容な方に向けた動画プレーヤーです。   

***
## 前提環境
 
 下記ソフトウェアがインストールされたWindows 11
 
* Lada Portable for Windows packed 250915
* [VLC media player 64bit版](https://www.videolan.org/vlc/index.ja.html)  

***
## インストール方法
 
* rp_pf.py　をLadaインストールフォルダにコピー 
* 追加パッケージ4つをインストール  
 Ladaインストールフォルダで、  
  
　`.\python\python.exe -m pip install PyQt6 PyOpenGL PyOpenGL-accelerate python-vlc`  

***
## 起動方法 

　Ladaインストールフォルダから   
  
　`.\python\python.exe rp_pf.py`
  
  ※VLC関連のエラーが出る方は、一旦、python-vlcをuninstallしてから起動してください。  
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
***<最大クリップ長>***が最も影響のある品質と速度のトレードオフ値です。8ぐらいから試してください。  
***<バッチサイズ>***は12にしておいてください。これ以上は効果ありません。  
再生済み場所や再生前のフレームをキャッシュすることでFPSを稼ぎます。  
お気に入りのシーンがある場合、その前あたりで一時停止し2-3分待つと本当のリアルタイム再生ができます。   

***
## 制限事項 

音は出ますが調整要です。  
JavPlayerEZを目指しましたがまだまだ足元にもおよびません。  
再生遅延、カクつき、音ズレが結構あります。  
モザイク検出モデルは v3.1 fast を固定で使用しています。  

***
## その他 

GPUリソースをフル活用できていません。 4070で使用率40%も使用していません。VRAMも4GB程度です。   
優秀なAI補佐とともにLADA処理をかなり調査しましたが力不足で改変までできておりません。  
すべてのソースが公開されていますので技術力がある方でしたら高速化可能です。 
ぜひ、このレポジトリで改良版をPRしてみてください。

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
