import os
import cv2
import time

def capture_images(interval, duration):
    # カメラのキャプチャを開始
    cap = cv2.VideoCapture(0)

    # カメラが正常に開かれたかどうかを確認
    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    # 写真を保存するディレクトリ
    output_dir = "./captured_images/"
    try:
        # ディレクトリが存在しない場合は作成する
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"ディレクトリの作成に失敗しました: {output_dir}")
        return

    start_time = time.time()
    image_count = 0

    while (time.time() - start_time) <= duration:
        ret, frame = cap.read()

        if not ret:
            print("フレームを取得できませんでした。")
            break

        # 画像をファイルに保存
        image_path = os.path.join(output_dir, f"image_{image_count}.jpg")
        cv2.imwrite(image_path, frame)

        image_count += 1
        time.sleep(interval)

    # キャプチャを解放して、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

# 関数を呼び出して写真を撮影する
capture_images(interval=2, duration=20)
