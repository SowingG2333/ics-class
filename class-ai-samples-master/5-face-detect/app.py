import cv2 as cv, time
from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    cap = cv.VideoCapture(0)
    while True:
        time.sleep(0.01) 
        return_value, frame = cap.read()

        image = cv.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)

