// gguf.go - minimal GGUF v3 reader (little-endian), stdlib only.
//
// Layout:
//   magic "GGUF" u32 | version u32 | tensor_count u64 | kv_count u64
//   kv entries: key(string) + value(type u32 + payload)
//   tensor infos: name(string) + n_dim u32 + dims[u64] + ggml_type u32 + offset u64
//   data section starts 32-byte aligned after the info section
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
)

const (
	ggmlF32 uint32 = 0
	ggmlF16 uint32 = 1
)

type tensorInfo struct {
	Name   string
	Ndim   uint32
	Dims   []uint64
	Type   uint32
	Offset uint64
}

type ggufFile struct {
	data  []byte // mmap of the whole file
	infos []tensorInfo
	align uint64 // data section absolute offset
}

func (g *ggufFile) u32(off uint64) uint32 {
	return binary.LittleEndian.Uint32(g.data[off : off+4])
}

func (g *ggufFile) u64(off uint64) uint64 {
	return binary.LittleEndian.Uint64(g.data[off : off+8])
}

func (g *ggufFile) str(off uint64) (string, uint64) {
	var n uint64 = g.u64(off)
	return string(g.data[off+8 : off+8+n]), off + 8 + n
}

func (g *ggufFile) skipValue(off uint64, vtype uint32) uint64 {
	switch vtype {
	case 0, 1: // u8, i8
		return off + 1
	case 2, 3: // u16, i16
		return off + 2
	case 4, 5, 6: // u32, i32, f32
		return off + 4
	case 7: // bool: 1 byte
		return off + 1
	case 8: // string
		var n uint64 = g.u64(off)
		return off + 8 + n
	case 10, 11, 12: // u64, i64, f64
		return off + 8
	case 9: // array: u32 elem_type, u64 count, count elements
		var elem uint32 = g.u32(off)
		var count uint64 = g.u64(off + 4)
		var p uint64 = off + 12
		var idx uint64 = 0
		for idx = 0; idx < count; idx++ {
			p = g.skipValue(p, elem)
		}
		return p
	}
	return off + 4
}

func loadGGUF(path string) (*ggufFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	g := &ggufFile{data: data}
	var off uint64 = 0
	var magic uint32 = g.u32(off)
	off += 4
	if magic != 0x46554747 {
		return nil, fmt.Errorf("not a GGUF file (magic %x)", magic)
	}
	off += 4 // version
	var ntensors uint64 = g.u64(off)
	off += 8
	var nkv uint64 = g.u64(off)
	off += 8
	var kv uint64 = 0
	for kv = 0; kv < nkv; kv++ {
		_, off = g.str(off)
		var vtype uint32 = g.u32(off)
		off += 4
		off = g.skipValue(off, vtype)
	}
	// tensor infos
	g.infos = make([]tensorInfo, 0, ntensors)
	var ti uint64 = 0
	for ti = 0; ti < ntensors; ti++ {
		var name string
		name, off = g.str(off)
		var ndim uint32 = g.u32(off)
		off += 4
		var dims []uint64 = make([]uint64, ndim)
		var d uint32 = 0
		for d = 0; d < ndim; d++ {
			dims[d] = g.u64(off)
			off += 8
		}
		var ttype uint32 = g.u32(off)
		off += 4
		var toff uint64 = g.u64(off)
		off += 8
		g.infos = append(g.infos, tensorInfo{Name: name, Ndim: ndim, Dims: dims, Type: ttype, Offset: toff})
	}
	// data section: 32-byte aligned after infos
	g.align = (off + 31) &^ 31
	return g, nil
}

// f16ToF32: IEEE 754 half -> float32 (5-bit exponent, bias 15).
// NOT the bf16 left-shift: that trick only works for bf16's 8-bit layout.
func f16ToF32(h uint16) float32 {
	var sign uint32 = uint32(h>>15) & 1
	var exp uint32 = uint32(h>>10) & 0x1F
	var man uint32 = uint32(h) & 0x3FF
	if exp == 0 {
		return math.Float32frombits((sign << 31) | man << 13) // subnormal
	}
	if exp == 0x1F {
		return math.Float32frombits((sign << 31) | 0x7F800000) // inf/nan -> inf
	}
	var e32 uint32 = exp - 15 + 127 // rebias
	return math.Float32frombits((sign << 31) | (e32 << 23) | (man << 13))
}

func (g *ggufFile) tensorF32(info *tensorInfo) []float32 {
	var nelem uint64 = 1
	var d uint32 = 0
	for d = 0; d < info.Ndim; d++ {
		nelem *= info.Dims[d]
	}
	var out []float32 = make([]float32, nelem)
	var base uint64 = g.align + info.Offset
	var idx uint64 = 0
	if info.Type == ggmlF32 {
		for idx = 0; idx < nelem; idx++ {
			out[idx] = math.Float32frombits(g.u32(base + idx*4))
		}
		return out
	}
	if info.Type == ggmlF16 {
		for idx = 0; idx < nelem; idx++ {
			var h uint16 = binary.LittleEndian.Uint16(g.data[base+idx*2 : base+idx*2+2])
			out[idx] = f16ToF32(h)
		}
		return out
	}
	return nil
}
