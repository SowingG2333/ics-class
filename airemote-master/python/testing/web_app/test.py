import cv2 as cv, time

class WebApp(object):

  def __init__(self, port=8000):
    self.port = port

  def run(self, **kwargs):
    from web_app import start_web
    from inferemote.testing.image_loader import ImageLoader
    try:
        cap = ImageLoader.get_stream(kwargs['data'])
        start_web(cap, self.port)
    except:
        pass

app = WebApp()
data = '/Users/haojiash/skiing.mp4'
app.run(data=data)
