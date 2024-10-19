from ctypes import *
import platform
from inferemote.ccbuf2numpy import Ccbuf2Numpy

class AirLib(object):
    ''' Airemote CC SharedObject Loading Helper '''
    MAX_BUFLEN = 9437184

    def __init__(self, libpath):
        dll = cdll.LoadLibrary
        self.lib = dll( "%s.%s" % (libpath, self.__get_so_ext()) )
        self.data_buf = create_string_buffer(self.MAX_BUFLEN)
        self.air = self.__create_air()

    def __mydel__(self):
        if self.lib and self.air:
            self.lib.destroy_air(c_int(self.air))

    def __get_so_ext(self):
        OS_PLATFORM = platform.system()
        if (OS_PLATFORM == 'Darwin'):
           so_ext = 'dylib'
        elif (OS_PLATFORM == 'Windows'):
           so_ext = 'dll'
        else:
           so_ext = 'so'
        return so_ext

    def __create_air(self):
        self.lib.create_air.restype = c_void_p
        return self.lib.create_air()

    def use_remote(self, remote):
        ''' c_char_p shares original buffer, so remote cannot be local variable. '''
        self.remote = remote.encode() # remote = b"tcp://localhost:5555"
        self.lib.use_remote.argtypes = (c_void_p, c_char_p)
        self.lib.use_remote(c_void_p(self.air), c_char_p(self.remote))

    def inference_remote(self, blob): 
        size_out = c_size_t(self.MAX_BUFLEN)
        size_buf = pointer(size_out)

        self.lib.inference_remote.restype = c_int
        rc = self.lib.inference_remote(c_void_p(self.air), \
             blob, c_int(len(blob)), self.data_buf, size_buf)
        # print("data_buf: ", data_buf.value)
        if rc:
            return None

        length = int.from_bytes(size_out, byteorder='little', signed=False)
        result = bytes(self.data_buf)[:length]

        #with open('output.bin', 'wb') as file:
        #    file.write(result)

        return Ccbuf2Numpy(bytes(result))

class AirObject(object):
    ''' Base Class for alorithm object such as googlenet/picasso '''

    def __init__(self, air=None):
        self.air = air

    def run(self, image):
        ''' This method will call pre_process and post_process methods of child class. '''
        input  = self.pre_process(image)
        output = self.air.inference_remote(input)
        result = self.post_process(output)
        return result

# Ends
