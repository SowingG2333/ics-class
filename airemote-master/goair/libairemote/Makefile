# Makefile

# 编译器和选项
CC = gcc
CFLAGS = -shared -fPIC

# 输出文件
LIBRARY = libairemote.so

# 源文件
SOURCES = airemoteapi.c

# 安装路径
INSTALL_PATH = /usr/local/lib

.PHONY: all install clean

all: $(LIBRARY)

$(LIBRARY):
	$(CC) $(CFLAGS) -o $@ $(SOURCES)

install:
	sudo mv $(LIBRARY) $(INSTALL_PATH)
	sudo ldconfig

clean:
	rm -f $(LIBRARY)
