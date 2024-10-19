package airlab

import (
	"image"
	"gocv.io/x/gocv"
	"goair/goair"
)

type Picasso struct {
	goair.AtlasRemote     // 远程推理模型
	iH, iW, oH, oW    int // 模型参数
}

func NewPicasso() *Picasso {
	p := &Picasso{iH: 720, iW: 1080, oH: 360, oW: 540}
	p.InitRemote(p)
	return p
}

func (p *Picasso) PreProcess(img gocv.Mat) []byte {
	blob := gocv.BlobFromImage(img, 1.0, image.Pt(p.iW, p.iH),
		gocv.NewScalar(0, 0, 0, 0), true, false)
	return blob.ToBytes()
}

func (p *Picasso) PostProcess(result [][]byte) interface{} {
	blob := result[0]
	prob, _ := gocv.NewMatWithSizesFromBytes([]int{3, p.oH, p.oW},
		gocv.MatTypeCV32FC3, blob)
	prob.ConvertTo(&prob, gocv.MatTypeCV8UC3)
	img := chw2hwc(prob)
	gocv.CvtColor(img, &img, gocv.ColorBGRToRGB)
	return img
}

func (p *Picasso) Inference(img gocv.Mat) gocv.Mat {
	return p.RunRemote(img).(gocv.Mat)
}

func chw2hwc(chw gocv.Mat) gocv.Mat {
	size := chw.Size()
	dims := size[1] * size[2]
	hwc := gocv.NewMatWithSize(size[1], size[2], gocv.MatTypeCV8UC3)
	for i := 0; i < dims; i++ {
		hwc.SetUCharAt(0, i*3, chw.GetUCharAt(0, i))          // R
		hwc.SetUCharAt(0, i*3+1, chw.GetUCharAt(0, i+dims))   // G
		hwc.SetUCharAt(0, i*3+2, chw.GetUCharAt(0, i+dims*2)) // B
	}
	return hwc
}
