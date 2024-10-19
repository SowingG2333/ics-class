package airlab

import (
	"fmt"
	"image"
	"sort"
	"unsafe"
	"gocv.io/x/gocv"
	"goair/goair"
)

type GoogleNet struct {
	goair.AtlasRemote     // 远程推理模型
	iH, iW      int // 模型参数
}

func NewGoogleNet() *GoogleNet {
	p := &GoogleNet{iH: 224, iW: 224}
	p.InitRemote(p)
	return p
}

func (g *GoogleNet) PreProcess(img gocv.Mat) []byte {
	blob := gocv.NewMat()
	gocv.Resize(img, &blob, image.Pt(g.iW, g.iH), 0, 0, gocv.InterpolationDefault)
	return blob.ToBytes()
}

func (g *GoogleNet) PostProcess(result [][]byte) interface{} {
	blob := result[0]
	data := make([]float32, len(blob)/4)
	for i := range data {
		data[i] = *(*float32)(unsafe.Pointer(&blob[i*4]))
	}
	topK := findTopKIndices(data, 5)
	fmt.Println("======== top5 inference results: =============")
	for _, n := range topK {
		fmt.Printf("label:%d  confidence: %f, class: %s\n", n, data[n], getImageNetClass(n))
	}
	return getImageNetClass(topK[0])
}

func (g *GoogleNet) Inference(img gocv.Mat) string {
	return g.RunRemote(img).(string)
}

func findTopKIndices(data []float32, topK int) []int {
	// Create a slice of index-value pairs.
	type IndexValuePair struct {
		Index int
		Value float32
	}
	pairs := make([]IndexValuePair, len(data))
	for i, v := range data {
		pairs[i] = IndexValuePair{i, v}
	}

	// Sort the pairs by value in descending order.
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Value > pairs[j].Value
	})

	// Extract the indices of the top K largest values.
	topKIndices := make([]int, topK)
	for i := 0; i < topK && i < len(pairs); i++ {
		topKIndices[i] = pairs[i].Index
	}

	return topKIndices
}
