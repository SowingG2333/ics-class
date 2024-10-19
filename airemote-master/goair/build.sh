#!/bin/bash

APP='gotest'
OPENCV_VERSION=$(opencv_version)

if [ ! -f ./go.mod ] || [ ! -f ./go.sum ]; then
	go mod init $APP
	# go mod tidy
	# gocv 和 opencv 的版本要对应，参考：https://pkg.go.dev/gocv.io/x/gocv
	if [ $OPENCV_VERSION == '4.2.0' ]; then
		go get gocv.io/x/gocv@v0.22.0
	elif [ $OPENCV_VERSION == '4.5.4' ]; then
		go get gocv.io/x/gocv@v0.29.0
	elif [ $OPENCV_VERSION == '4.5.5' ]; then
		go get gocv.io/x/gocv@v0.30.0
	elif [ $OPENCV_VERSION == '4.9.0' ]; then
		go get gocv.io/x/gocv@v0.36.1
	fi
fi

go build -o $APP
# go run *.go
