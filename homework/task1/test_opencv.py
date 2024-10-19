import cv2
from flask import Flask, render_template, Response, request

app = Flask(__name__)

# 加载人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames(source):
    if source == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    while True:
        # 捕获帧
        success, frame = cap.read()
        if not success:
            break

        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 在检测到的人脸上应用模糊处理
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y+h, x:x+w] = face_region
        
        # 编码帧为 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # 生成视频流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    source = request.args.get('source', default='camera')
    return Response(generate_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)