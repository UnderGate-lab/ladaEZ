1.前提環境

　Lada Portable for Windows packed 250915

2.インストール方法

　rp_pf.py　をLadaインストールフォルダにコピー

　追加パッケージ3つをインストール　

　Ladaインストールフォルダで、
　
　.\python\python.exe -m pip install PyQt6 PyOpenGL PyOpenGL-accelerate

3.起動方法

　Ladaインストールフォルダから

　.\python\python.exe rp_pf.py

4.操作方法
　
　停止再開：　　　　画面クリック、SPACE
　再生位置移動：　　進捗バーをクリック
　フルスクリーン：　ダブルクリック、F

　設定のバッチフレーム数は12にしておいてください。
　現在はこれ以上だと逆にオーバーヘッドとなります。

5.制限事項

　音は出ません。今後対応予定。
　最大15FPS程度です。（RTX 2060以上ならどれでも違いなし）

　Ladaの内部処理を変更して処理速度改善中です。乞うご期待！

