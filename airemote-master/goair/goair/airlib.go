package goair

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L/usr/local/lib -lairemote
#include "airemoteapi.h"
*/
import "C"

import (
	"encoding/binary"
	"unsafe"
)

type AirLib struct {
	MAX_BUFFER C.size_t
	dataBuf    []byte
	air        unsafe.Pointer
	remote     *C.char
}

func (al *AirLib) initAirlib() {
	al.MAX_BUFFER = 9437184
	al.dataBuf = make([]byte, al.MAX_BUFFER)
	al.air = C.create_air()
}

func NewAirlib() *AirLib {
	al := &AirLib{}
	al.initAirlib()
	return al
}

func (al *AirLib) Close() {
	C.destroy_air(al.air)
}

func (al *AirLib) UseRemote(remote string) {
	al.remote = C.CString(remote)
	C.use_remote(al.air, al.remote)
}

func (al *AirLib) InferenceRemote(blob []byte) [][]byte {
	length := al.MAX_BUFFER
	C.inference_remote(al.air, unsafe.Pointer(&blob[0]), C.size_t(len(blob)),
		unsafe.Pointer(&al.dataBuf[0]), (*C.size_t)(unsafe.Pointer(&length)))
	result := al.dataBuf[:length]
	// fmt.Printf("length: %d\nresult: %v\n", length, result)
	return ccbuf2slice(result)
}

func ccbuf2slice(result []byte) [][]byte {
	outputsNum := binary.LittleEndian.Uint32(result[:4])
	// outputsSize := binary.LittleEndian.Uint32(result[4:8])

	offsets := make([]uint32, outputsNum)
	sizes := make([]uint32, outputsNum)

	for i := uint32(0); i < outputsNum; i++ {
		offsets[i] = binary.LittleEndian.Uint32(result[8+i*8 : 12+i*8])
		sizes[i] = binary.LittleEndian.Uint32(result[12+i*8 : 16+i*8])
	}

	resultsList := make([][]byte, outputsNum)
	for i := uint32(0); i < outputsNum; i++ {
		blob := result[offsets[i] : offsets[i]+sizes[i]]
		resultsList[i] = blob
	}

	return resultsList
}

func slice2ccbuf(resultsList [][]byte) []byte {
	outputsNum := uint32(len(resultsList))
	outputsSize := uint32(0)
	bytesList := make([][]byte, outputsNum)

	for i := uint32(0); i < outputsNum; i++ {
		result := resultsList[i]
		bytesList[i] = result
		outputsSize += uint32(len(result))
	}

	blob := make([]byte, 0)
	blob = append(blob, make([]byte, 4)...)
	binary.LittleEndian.PutUint32(blob[:4], outputsNum)
	binary.LittleEndian.PutUint32(blob[4:8], outputsSize)

	offset := uint32(8 * (outputsNum + 1))
	for i := uint32(0); i < outputsNum; i++ {
		binary.LittleEndian.PutUint32(blob[len(blob):len(blob)+4], offset)
		size := uint32(len(bytesList[i]))
		binary.LittleEndian.PutUint32(blob[len(blob):len(blob)+4], size)
		offset += size
	}

	for i := uint32(0); i < outputsNum; i++ {
		blob = append(blob, bytesList[i]...)
	}

	return blob
}
