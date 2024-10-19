package goair

import (
	"gocv.io/x/gocv"
)

type Process interface {
	PreProcess(gocv.Mat) []byte
	PostProcess([][]byte) interface{}
}

type AtlasRemote struct {
	AirLib
	proc Process
}

func (ar *AtlasRemote) InitRemote(p Process) {
	ar.initAirlib()
	ar.proc = p
}

func (ar *AtlasRemote) RunRemote(img gocv.Mat) interface{} {
	input := ar.proc.PreProcess(img)
	output := ar.InferenceRemote(input)
	result := ar.proc.PostProcess(output)
	return result
}
