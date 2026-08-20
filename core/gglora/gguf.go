// gguf.go - minimal GGUF v3 reader (little-endian), stdlib only.
//
// Layout:
//
//	magic "GGUF" u32 | version u32 | tensor_count u64 | kv_count u64
//	kv entries: key(string) + value(type u32 + payload)
//	tensor infos: name(string) + n_dim u32 + dims[u64] + ggml_type u32 + offset u64
//	data section starts 32-byte aligned after the info section
package main

import (
	"encoding/binary"
	"fmt"
	"io/fs"
	"math"
	"os"
	"syscall"
)

const (
	ggmlF32  uint32 = 0
	ggmlF16  uint32 = 1
	ggmlQ8_0 uint32 = 8
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
	align uint64         // data section absolute offset
	kv    map[string]any // metadata key-values (strings/numbers/arrays)
}

// kvString / kvInt / kvFloat / kvArray: typed metadata accessors.
func (g *ggufFile) kvString(key string) string {
	var v, ok = g.kv[key].(string)
	if ok {
		return v
	}
	return ""
}

func (g *ggufFile) kvInt(key string) int {
	var v, ok = g.kv[key].(int)
	if ok {
		return v
	}
	{
		var v, ok = g.kv[key].(int64)
		if ok {
			return int(v)
		}
	}
	return 0
}

func (g *ggufFile) kvFloat(key string) float64 {
	var v, ok = g.kv[key].(float64)
	if ok {
		return v
	}
	{
		var v, ok = g.kv[key].(int)
		if ok {
			return float64(v)
		}
	}
	{
		var v, ok = g.kv[key].(int64)
		if ok {
			return float64(v)
		}
	}
	return 0
}

func (g *ggufFile) kvArray(key string) []any {
	var v, ok = g.kv[key].([]any)
	if ok {
		return v
	}
	return nil
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
	var f, err = os.Open(path)
	if err != nil {
		return nil, err
	}
	var st fs.FileInfo

	st, err = f.Stat()
	if err != nil {
		return nil, err
	}
	var data []byte

	data, err = syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	var g = &ggufFile{data: data, kv: map[string]any{}}
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
		var key string
		key, off = g.str(off)
		var vtype uint32 = g.u32(off)
		off += 4
		var v any
		v, off = g.readValue(off, vtype)
		g.kv[key] = v
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

// readValue: parse one KV value (returns the decoded Go value and the next
// offset). Arrays decode to []any.
func (g *ggufFile) readValue(off uint64, vtype uint32) (any, uint64) {
	switch vtype {
	case 0:
		return int(g.data[off]), off + 1
	case 1:
		return int(int8(g.data[off])), off + 1
	case 2:
		return int(binary.LittleEndian.Uint16(g.data[off : off+2])), off + 2
	case 3:
		return int(int16(binary.LittleEndian.Uint16(g.data[off : off+2]))), off + 2
	case 4:
		return int(binary.LittleEndian.Uint32(g.data[off : off+4])), off + 4
	case 5:
		return int(int32(binary.LittleEndian.Uint32(g.data[off : off+4]))), off + 4
	case 6:
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(g.data[off : off+4]))), off + 4
	case 7:
		return g.data[off] != 0, off + 1
	case 8:
		var n uint64 = g.u64(off)
		return string(g.data[off+8 : off+8+n]), off + 8 + n
	case 10:
		return int64(binary.LittleEndian.Uint64(g.data[off : off+8])), off + 8
	case 11:
		return int64(int64(binary.LittleEndian.Uint64(g.data[off : off+8]))), off + 8
	case 12:
		return math.Float64frombits(binary.LittleEndian.Uint64(g.data[off : off+8])), off + 8
	case 9: // array
		var elem uint32 = g.u32(off)
		var count uint64 = g.u64(off + 4)
		var p uint64 = off + 12
		var arr []any = make([]any, 0, count)
		var idx uint64 = 0
		for idx = 0; idx < count; idx++ {
			var v any
			v, p = g.readValue(p, elem)
			arr = append(arr, v)
		}
		return arr, p
	}
	return nil, off + 4
}

// f16ToF32: IEEE 754 half -> float32 (5-bit exponent, bias 15).
// NOT the bf16 left-shift: that trick only works for bf16's 8-bit layout.
func f16ToF32(h uint16) float32 {
	var sign uint32 = uint32(h>>15) & 1
	var exp uint32 = uint32(h>>10) & 0x1F
	var man uint32 = uint32(h) & 0x3FF
	if exp == 0 {
		return math.Float32frombits((sign << 31) | man<<13) // subnormal
	}
	if exp == 0x1F {
		return math.Float32frombits((sign << 31) | 0x7F800000) // inf/nan -> inf
	}
	var e32 uint32 = exp - 15 + 127 // rebias
	return math.Float32frombits((sign << 31) | (e32 << 23) | (man << 13))
}

// unmap: release the model-file mmap. The dequantized weights live in RAM;
// the file mapping is only needed while reading tensors. A long-lived
// trainer must not pin the whole file.
func (g *ggufFile) unmap() {
	if g.data != nil {
		_ = syscall.Munmap(g.data)
		g.data = nil
	}
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
	if info.Type == ggmlQ8_0 {
		// Q8_0 block: f16 scale d (2 bytes), then 32 int8 values.
		// x[i] = q[i] * d. (llama.cpp: block = {ggml_half d; int8 qs[32]},
		// 34 bytes per block - NOT 36.)
		var block uint64 = 0
		for block = 0; block < nelem/32; block++ {
			var d float32 = f16ToF32(binary.LittleEndian.Uint16(g.data[base+block*34 : base+block*34+2]))
			var qbase uint64 = base + block*34 + 2
			var i uint32 = 0
			for i = 0; i < 32; i++ {
				out[block*32+uint64(i)] = float32(int8(g.data[qbase+uint64(i)])) * d
			}
		}
		if nelem%32 != 0 {
			return nil // ragged Q8_0 is not expected for weight tensors
		}
		return out
	}
	return nil
}

// findTensor: locate a tensor by exact name.
func (g *ggufFile) findTensor(name string) *tensorInfo {
	var i int = 0
	for i = 0; i < len(g.infos); i++ {
		if g.infos[i].Name == name {
			return &g.infos[i]
		}
	}
	return nil
}
