// http.go - shared HTTP transport and buffer pools for the gateway.
//
// Leak discipline: every outbound call goes through one client with HARD
// timeouts. A hung llama backend must never pin a goroutine, a connection,
// or a file descriptor forever (the old http.DefaultClient had no timeout
// at all). The 1MB SSE/ledger scanner allocations are pooled instead of
// re-allocated per request.
package main

import (
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

var httpClient = &http.Client{
	Timeout: 120 * time.Second,
	Transport: &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:        32,
		MaxIdleConnsPerHost: 8,
		IdleConnTimeout:     60 * time.Second,
	},
}

// httpPostJSON: POST JSON, return the response body. Used by the chain
// summarizer and the translation bridge.
func httpPostJSON(url string, body []byte) ([]byte, error) {
	var req, err = http.NewRequest("POST", url, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	var resp *http.Response

	resp, err = httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

// httpClientSlow: for genuinely long operations (the tool loop, context
// compaction) where a contended potato can exceed the 120s interactive cap.
var httpClientSlow = &http.Client{
	Timeout: 600 * time.Second,
	Transport: &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:        16,
		MaxIdleConnsPerHost: 4,
		IdleConnTimeout:     60 * time.Second,
	},
}

// httpPostJSONSlow: POST with the long timeout (tool loop / compaction).
func httpPostJSONSlow(url string, body []byte) ([]byte, error) {
	var req, err = http.NewRequest("POST", url, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	var resp *http.Response

	resp, err = httpClientSlow.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

// ---- 1MB scanner buffer pool ---------------------------------------------
// bufio.Scanner needs a buffer big enough for the longest line (a huge
// streamed SSE frame or a giant ledger turn). Allocating 1MB per request
// is churn; pool it. The buffer must not be returned until the Scanner
// loop has finished (the Scanner references it during Scan).

var bufPool = sync.Pool{
	New: func() any {
		var b []byte = make([]byte, 1<<20)
		return &b
	},
}

func getScanBuf() []byte {
	return *bufPool.Get().(*[]byte)
}

func putScanBuf(b []byte) {
	if cap(b) >= 1<<20 {
		bufPool.Put(&b)
	}
}
