"""
Inferemote: a Remote Inference Toolkit for Ascend 310

Copyright (c) 2021 Jiasheng Hao <haojiash@qq.com>
(University of Electronic Science and Technology of China, UESTC)

Permission  is  hereby  granted,  free  of  charge,  to  any  person
obtaining a copy of this software and associated documentation files
(the  "Software"),  to deal  in  the  Software without  restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to  permit persons to whom  the Software is furnished  to do so,
subject to the following conditions:

The  above copyright  notice  and this  permission  notice shall  be
included in all copies or substantial portions of the Software.

THE  SOFTWARE IS  PROVIDED "AS  IS", WITHOUT  WARRANTY OF  ANY KIND,
EXPRESS OR IMPLIED,  INCLUDING BUT NOT LIMITED TO  THE WARRANTIES OF
MERCHANTABILITY,    FITNESS   FOR    A   PARTICULAR    PURPOSE   AND
NONINFRINGEMENT. IN NO EVENT SHALL  THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR  ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN
ACTION OF  CONTRACT, TORT OR OTHERWISE,  ARISING FROM, OUT OF  OR IN
CONNECTION WITH  THE SOFTWARE OR  THE USE  OR OTHER DEALINGS  IN THE
SOFTWARE.

"""
''' This upgrade costed quite a couple of hours. I was to make a single 
module for showing the results in many web browsers for students from 
different hosts. So pub/sub mode of pyzmq should work for this. Many tries
fail since using a TCP/port was not a good idea for this. What was learned
is that Flask with monkey patching for pywsgi just won't work. In the end,
Fastapi showes a good solution.
'''
import os, time
import cv2 as cv
from fastapi import Body, FastAPI, Form, Request,Header,Response
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, threading, webbrowser

''' Exports '''
app = FastAPI()

cap = None

''' Callback function from LivewebTest'''
def start_web(_cap, port=8000):
    global cap
    cap = _cap

    def open_browser():
        time.sleep(1)
        webbrowser.open("http://localhost:%s" % port)

    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host='0.0.0.0', port=port, log_level="info")

def _gen():
    while True:
        success, frame = cap.read()
        #time.sleep(0.005)
        if success: 
            image = cv.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

''' Internal definitions '''
'''
origins = [ "*" ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
'''

file_path = os.path.dirname(os.path.realpath(__file__))
templates = Jinja2Templates(directory=os.path.join(file_path, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(file_path, "static")), name="static")
print('2222222222')

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})

@app.get('/video_feed', response_class=HTMLResponse)
async def video_feed():
    return  StreamingResponse(_gen(),
                    media_type='multipart/x-mixed-replace; boundary=frame')

