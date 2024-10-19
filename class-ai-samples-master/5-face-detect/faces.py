import cv2 as cv, time
from flask import Flask, render_template, Response

app = Flask(__name__)

'''
  See what models there are: python -c 'import cv2;print(cv2.data.haarcascades)' 
'''
#face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route('/')
def index():
    return render_template('index.html')

def dectect_face(image):
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x,y,w,h) in faces:
          image = cv.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 4)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image

def gen():
    cap = cv.VideoCapture(0)
    while True:
        time.sleep(0.01) 
        return_value, frame = cap.read()

        faces = dectect_face(frame)

        image = cv.imencode('.jpg', faces)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

'''
https://www.codenong.com/3cdafb737d2c16fbaa51/
'''
