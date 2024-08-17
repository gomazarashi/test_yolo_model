from ultralytics import YOLO
import sys

# ファイルパス取得
file_path = sys.argv[1]

# モデル読み込み
model = YOLO("best.pt")

# 入力画像
results = model(file_path)
print(len(results[0].boxes))

print("物体検出と保存が完了しました。")
