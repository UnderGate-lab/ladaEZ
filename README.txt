1.前提環境

　Lada Portable for Windows packed 250915

2.インストール方法

　rp_pf.py　をLadaインストールフォルダにコピー

　追加パッケージ4つをインストール　

　Ladaインストールフォルダで、
　
　.\python\python.exe -m pip install PyQt6 PyOpenGL PyOpenGL-accelerate python-vlc

3.起動方法

　Ladaインストールフォルダから

　.\python\python.exe rp_pf.py

4.操作方法
　
　停止再開：　　　　画面クリック、SPACE
　再生位置移動：　　進捗バーをクリック
　フルスクリーン：　ダブルクリック、F
　など

　設定のバッチフレーム数は12にしておいてください。
　クリップ長が品質と速度のトレードオフ値です。

5.制限事項

　音は出ますが、まだ調整要です。



