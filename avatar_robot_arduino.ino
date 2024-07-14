#include <Servo.h>

//ピン定義
const int leftEyeLEDPin = 3;
const int rightEyeLEDPin = 2;
const int smileLEDPin1 = 5;
const int smileLEDPin2 = 6;
const int smileLEDPin3 = 7;
const int servoPin = 9;

//サーボモータのインスタンスを作成
Servo myServo;

void setup() {
  //ピンモードの設定
  pinMode(leftEyeLEDPin, OUTPUT);
  pinMode(rightEyeLEDPin, OUTPUT);
  pinMode(smileLEDPin1, OUTPUT);
  pinMode(smileLEDPin2, OUTPUT);
  pinMode(smileLEDPin3, OUTPUT);

  //サーボの初期化
  myServo.attach(servoPin);

  //シリアル通信の初期化
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    //シリアル通信からデータを読み取る
    String data = Serial.readStringUntil('\n'); 
    int neck_angle, left_blink, right_blink, smile; //Pythonで生成したシリアル通信を受信する変数
    
    //読み取ったデータを解析する
    sscanf(data.c_str(), "%d,%d,%d,%d", &neck_angle, &left_blink, &right_blink, &smile);

    //左目のLED制御
    if (left_blink == 0){
      digitalWrite(leftEyeLEDPin, HIGH); //目を開けている(0)なら光る
    } else {
      digitalWrite(leftEyeLEDPin, LOW);
    }
    
    //右目のLED制御
    if (right_blink == 0){
      digitalWrite(rightEyeLEDPin, HIGH); //目を開けている(0)なら光る
    } else {
      digitalWrite(rightEyeLEDPin, LOW);
    }

    //笑顔に応じたLEDの制御
    if (smile ==1){ //笑顔の閾値が1.0のとき，LEDを3つ光らせる
      digitalWrite(smileLEDPin1, HIGH);
      digitalWrite(smileLEDPin2, HIGH);
      digitalWrite(smileLEDPin3, HIGH);
    } else if (smile >0.6){ //笑顔の閾値が0.6~0.9のとき，LEDを2つ光らせる
      digitalWrite(smileLEDPin1, HIGH);
      digitalWrite(smileLEDPin2, LOW);
      digitalWrite(smileLEDPin3, HIGH);
    } else if (smile >0.2){ //笑顔の閾値が0.2~0.5のとき，LEDを1つ光らせる
      digitalWrite(smileLEDPin1, LOW);
      digitalWrite(smileLEDPin2, HIGH);
      digitalWrite(smileLEDPin3, LOW);
    } else { //笑顔の閾値が~0.2のとき，LEDは光らない
      digitalWrite(smileLEDPin1, LOW);
      digitalWrite(smileLEDPin2, LOW);
      digitalWrite(smileLEDPin3, LOW);
    }

    //(首の上下)サーボモータの制御
    if (neck_angle == 1) {
      myServo.write(170);  //上を向いているとき
    } else if(neck_angle == 1) {
      myServo.write(10);  //正面を向いているとき
    } else 
      myServo.write(190);  //バグ
  }
}
