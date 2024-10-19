from airlib import AirLib

air = AirLib('libgooglenet')
air.use_remote('tcp://localhost:5530')
#air.use_remote('dummy://./googlenet/googlenet-dummy.bin')

blob = open('rabit.jpg', 'rb').read()
text = air.inference_remote(blob)

print(f"\nResult: ``{text}''\n")

