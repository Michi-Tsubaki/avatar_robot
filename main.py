#機械工学総合演習第二　アドバンストプログラミング課題
##テーマ: 簡単な1自由度アバターロボットの作成
##製作者: 03240281 機械情報工学科3年 椿道智(Michitoshi TSUBAKI)
##製作期間: 2024/7/13~

#概要: 首の角度と瞬きと笑顔を検出し，シリアル通信でその情報をArduinoに送信するプログラムを作成する．

#仕様
#首の角度は，顔の中心と首の中心の角度を計算することで求める．サーボモータを用いて首を動かすからシリアル通信の値は0~180の間になる．
##首の角度はopencvとdlibを用いて頭部方向推定を行うことで実装する．

#瞬きは，目の開閉の比率を計算することで求める．LEDを光らせるが，光るか光らないかなので，シリアル通信の値は0か1になる．
##opencvを用いた瞬きの検出は分類器「haarcascade_eye_tree_eyeglasses.xml」を用いて実装する．

#笑顔は，笑顔か笑顔でないかを判定する．LEDをシリアル通信の値は0か1になる．
##opencvを用いた笑顔検出は分類器「haarcascade_frontalface_default.xml」と分類器「haarcascade_smile.xml」を用いて実装する．

#シリアル通信は，pyserialを用いて実装する．
##シリアル通信の値は，首の角度，瞬き，笑顔の順に送信する．
############################################################################################################

#必要なライブラリのインポート
import cv2 #画像処理用ライブラリ（OpenCV）
import numpy as np #数値計算用ライブラリ（Numpy）
import serial #Arduinoにシリアル通信を行うためのライブラリ
import time
import serial.tools.list_ports
from ultralytics import YOLO

#シリアル通信のポート番号を表示
ports = serial.tools.list_ports.comports()
for port in ports:
    print(port.device)

com_num = "COM3"

#シリアル通信の設定
ser = serial.Serial(com_num, 9600) # Arduinoのポートを指定

#分類器の読み込み
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#YOLOのモデル(重みファイル)を指定する(あらかじめ学習済みの重みファイルを用意しておく)
model = YOLO("last.pt")

# カメラの設定
cap = cv2.VideoCapture(0)  # カメラを指定
#メインのループ
while True:
    #カメラからフレームを取得
    ret, frame = cap.read() #retはフレームが取得できたかどうかの真偽値，frameは取得したフレーム 
    if not ret:
        break #フレームが取得できなかった場合は強制終了

    #YOLOで顔の角度を検出
    result = model.predict(frame)
    r = result[0]
    r.save_txt("result.txt", save_conf=True)
    f = open('result.txt', 'r', encoding='UTF-8')
    data = f.read()
    f.close()
    lines = data.split('\n')
    num_lines = len(lines)
    for line in lines[:-1]:
        if len(line.split()) == 6:
            obj, x_center, y_center, width, height, cof = map(float, line.split())
    neck_angle = obj
    #顔の角度を表示(デバッグ用)
    label = "Neck Angle " + str(neck_angle)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.putText(frame, label, (20,50), font, font_scale, (0, 0, 0), font_thickness)

    #フレームをグレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #顔検出の分類器を作用
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    if len(faces) == 1 or len(faces) == 2: #バグ防止のため，顔が1つ検出された場合に限定して瞬き処理をする．
        for (x, y, w, h) in faces:
            #顔の中心と顎の中心を計算
            face_center = (x + w // 2, y + h // 2) 
            chin_center = (x + w // 2, y + h)
            #顔領域を描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            eyes_gray = gray[y : y + int(h/2), x : x + w]
            eyes = eye_cascade.detectMultiScale(eyes_gray)

            left_blink = 1 #左の瞬き(初期値1(瞬きあり))
            right_blink = 1 #右の瞬き(初期値1(瞬きあり))

            if len(eyes) == 1 or len(eyes) == 2: #バグ防止のため，目が1つか2つ検出された場合に限定して瞬き処理をする．
                for (ex, ey, ew, eh) in eyes: #目の絶対座標を取得
                    ex += x
                    ey += y
                    eye_center = (ex + ew // 2, ey + eh // 2)
                    if eye_center[0] < x + w // 2: #左目の場合
                        left_blink = 1 if eh/ew < 0.25 else 0 #左目の瞬きを検出
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3) #デバッグのため描画
                        label = "Left " + str(left_blink)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        font_thickness = 2
                        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                        label_position = (ex, ey)  # 円の中心から少し下にラベルを配置
                        cv2.putText(frame, label, label_position, font, font_scale, (0, 255, 0), font_thickness)
                    else:
                        right_blink = 1 if eh/ew < 0.25 else 0
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
                        label = "Right " + str(right_blink)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        font_thickness = 2
                        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                        label_position = (ex, ey)  # 円の中心から少し下にラベルを配置
                        cv2.putText(frame, label, label_position, font, font_scale, (0, 255, 0), font_thickness)

        #笑顔を検出 (参考: 「PythonとOpenCVを使った笑顔認識」https://qiita.com/fujino-fpu/items/99ce52950f4554fbc17d）
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray,(100,100))
        #輝度で規格化
        lmin = roi_gray.min() #輝度の最小値
        lmax = roi_gray.max() #輝度の最大値
        for index1, item1 in enumerate(roi_gray):
            for index2, item2 in enumerate(item1):
                roi_gray[index1][index2] = int((item2 - lmin)/(lmax-lmin) * item2)
        smiles= smile_cascade.detectMultiScale(roi_gray,scaleFactor= 1.1, minNeighbors=0, minSize=(20, 20)) #笑顔識別
        if len(smiles) >0 : #笑顔領域がなければ以下の処理をパス(in order to 高速化)
            smile_neighbors = len(smiles) #サイズを考慮した笑顔認識
            LV = 2/100 #笑顔の強度
            intensityZeroOne = smile_neighbors  * LV
            if intensityZeroOne > 1:
                intensityZeroOne = 1
            smile = intensityZeroOne
            
            #笑顔指数(0~4):Arduinoに送信する値を表示(デバッグ用)
            label = "Smile " + str(smile)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.putText(frame, label, (100,100), font, font_scale, (0, 0, 0), font_thickness)

            #笑顔を描画:笑顔の強度に応じて円の大きさを変える(デバッグ用)
            for(sx,sy,sw,sh) in smiles:
                cv2.circle(frame,(int(x+(sx+sw/2)*w/100),int(y+(sy+sh/2)*h/100)),int(sw/2*w/100), (255*(1.0-intensityZeroOne), 0, 255*intensityZeroOne),1)

        # シリアル通信でデータを送信
        print(f'{int(neck_angle)},{left_blink},{right_blink},{smile}')
        ser.write(f'{int(neck_angle)},{left_blink},{right_blink},{smile}\n'.encode())

    #「デバッグ用画面」というラベルを表示
    label = "For Debug"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.putText(frame, label, (20,20), font, font_scale, (0, 0, 0), font_thickness)

    #フレームを表示(デバッグ用)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Arduinoに終了信号を送信
        print('End')
        #ser.write('-1,-1,-1,-1\n'.encode())
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
ser.close()