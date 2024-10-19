package main

import (
    "fmt"
    "gocv.io/x/gocv"
    "goair/airlab"
)

func test_googlenet() {
    air := airlab.NewGoogleNet()
    //air.UseRemote("tcp://localhost:5530")
    air.UseRemote("dummy://./googlenet.bin")

    img := gocv.IMRead("./test.jpg", gocv.IMReadColor)
    out := air.Inference(img)

    fmt.Println("Result: " + out)
}

func test_picasso() {
    air := airlab.NewPicasso()
    //air.UseRemote("tcp://localhost:5550")
    air.UseRemote("dummy://./picasso.bin")

    img := gocv.IMRead("./test.jpg", gocv.IMReadColor)
    out := air.Inference(img)

    gocv.IMWrite("result.jpg", out)

    //window := gocv.NewWindow("Result")
    //window.IMShow(out)
    //window.WaitKey(0)
}

func main() {
    test_googlenet()
    test_picasso()
}
