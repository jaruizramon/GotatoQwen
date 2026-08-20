// heap.go - the manual heap: one fixed float32 arena, bump-allocated,
// explicitly freed. This is the "no GC" contract in code.
//
// GOGC=off means the runtime NEVER collects, so every make() on the
// training hot path would grow the process heap permanently (the per-
// sample LoRA h vectors, the dyh backward rows and dXHead were ~100MB of
// unreachable garbage per sample). The manual heap replaces those makes
// with a single arena: heapAllocF32 bump-allocates (zeroed), heapReset
// frees everything since the last reset. The programmer owns the heap:
// lifetimes are explicit (reset at the sample boundary), capacity is
// explicit (heapInit sized for the worst sample), and an exhausted arena
// panics like a real malloc abort instead of silently corrupting the math.
package main

import "strconv"

// mheap: the one arena. A package-level singleton: the trainer has exactly
// one hot path at a time (train or collect), and the arena lives for the
// whole process, so its memory stays flat for the entire run.
var mheap manualHeap

type manualHeap struct {
	buf  []float32
	used int // bump pointer: next free slot
}

// heapInit: (re)size the arena to `capacity` floats. Every allocation
// granted before the call is invalidated.
func heapInit(capacity int) {
	mheap.buf = make([]float32, capacity)
	mheap.used = 0
}

// heapReset: explicit free of every allocation since the last reset.
// The bump pointer returns to zero; the memory itself is not zeroed
// (every allocation is zeroed on grant, and writers overwrite what they
// use). Call once per sample, before the sample's forward pass.
func heapReset() {
	mheap.used = 0
}

// heapAllocF32: bump-allocate n zeroed floats. Panics when the arena is
// exhausted - the caller sized heapInit wrong (worst sample + reset
// discipline), and returning stale memory would corrupt the training.
func heapAllocF32(n int) []float32 {
	if n <= 0 {
		return nil
	}
	if mheap.used+n > len(mheap.buf) {
		panic("manual heap exhausted: need " + strconv.Itoa(n) + " floats, " +
			strconv.Itoa(mheap.used) + "/" + strconv.Itoa(len(mheap.buf)) + " used")
	}
	var out []float32 = mheap.buf[mheap.used : mheap.used+n]
	mheap.used += n
	var i int = 0
	for i = 0; i < n; i++ {
		out[i] = 0
	}
	return out
}

// heapBytes: arena capacity in bytes (for the [mem] report).
func heapBytes() int {
	return len(mheap.buf) * 4
}
