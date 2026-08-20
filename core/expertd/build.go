// build.go - the Go expert builder (replaces expert-builder.py).
//
//	expertd build <lang> <stack>
//
// Spawned by `expertd watch` (nice 10, detached). Collects the stack's
// <lang> corpus, trains the LoRA adapter with gglora (the Go trainer -
// no python anywhere), and publishes the result to index.json. Appends
// progress to build_<lang>.log. Exit 0 = ready, 1 = error.
//
// The exact collect rules of the retired python builder are preserved:
// walk the stack, extension filter, 80..300KB size window, sha256 dedupe,
// 40-file / 1MB caps, UTF-8-with-replacement decoding.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const maxCorpusFiles int = 40
const maxCorpusBytes int = 1_000_000

func buildCmd(args []string) {
	if len(args) < 2 {
		fmt.Println("usage: expertd build <lang> <stack>")
		os.Exit(2)
	}
	var lang string = args[0]
	var stack string = args[1]
	var logPath = fleetDir + "/build_" + lang + ".log"
	var logFile, err = os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		os.Exit(1)
	}
	defer logFile.Close()
	var logf = func(format string, a ...any) {
		var line = fmt.Sprintf(format, a...) + "\n"
		_, _ = fmt.Fprint(logFile, line)
		_, _ = fmt.Fprint(os.Stderr, line)
	}

	var t0 time.Time = time.Now()
	logf("[builder:%s] collect from %s ...", lang, stack)
	var corpusPath string

	var nfiles int

	var nbytes int

	corpusPath, nfiles, nbytes, err = collectCorpus(lang, stack)
	if err != nil {
		publishBuild(lang, false, "", 0, "collect failed: "+err.Error())
		logf("[builder:%s] COLLECT FAILED: %v", lang, err)
		os.Exit(1)
	}
	logf("[builder:%s] corpus: %d files, %d bytes", lang, nfiles, nbytes)

	var adapter string = fleetDir + "/adapters/" + lang + ".gguf"
	var base string = fleetDir + "/Qwen3-0.6B-Q8_0.gguf"
	logf("[builder:%s] training LoRA (gglora, 4 threads, window 128) ...", lang)
	var cmd = exec.Command(ggloraBin, "train", "--base", base,
		"--corpus", corpusPath, "--out", adapter,
		"--threads", "4", "--window", "128", "--name", lang)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	var runErr error = cmd.Run()
	var dt float64 = time.Since(t0).Seconds()
	if runErr != nil {
		publishBuild(lang, false, "", 0, "train failed: "+runErr.Error())
		logf("[builder:%s] TRAIN FAILED: %v", lang, runErr)
		os.Exit(1)
	}
	{
		var st, err = os.Stat(adapter)
		if err != nil || st.Size() == 0 {
			publishBuild(lang, false, "", 0, "train failed: no adapter written")
			logf("[builder:%s] TRAIN FAILED: no adapter written", lang)
			os.Exit(1)
		}
	}
	logf("[builder:%s] READY -> adapters/%s.gguf in %.0fs", lang, lang, dt)
	publishBuild(lang, true, "adapters/"+lang+".gguf", dt, "")
	recordStackManifest(lang, stack, true)
	os.Exit(0)
}

// collectCorpus: walk the stack for <lang> files and write corpus/<lang>.txt.
func collectCorpus(lang string, stack string) (string, int, int, error) {
	var corpusDir string = fleetDir + "/corpus"
	var err = os.MkdirAll(corpusDir, 0755)
	if err != nil {
		return "", 0, 0, err
	}
	var out string = corpusDir + "/" + lang + ".txt"
	var seen map[[32]byte]bool = make(map[[32]byte]bool)
	var files []string
	var errStop error = errors.New("stop")
	var walkErr error = filepath.Walk(stack, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			if info.Name() == ".git" || info.Name() == "node_modules" || info.Name() == "__pycache__" {
				return filepath.SkipDir
			}
			return nil
		}
		if extToLang[strings.ToLower(filepath.Ext(p))] != lang {
			return nil
		}
		var data []byte

		data, err = os.ReadFile(p)
		if err != nil {
			return nil
		}
		if len(data) < 80 || len(data) > 300*1024 {
			return nil
		}
		var h [32]byte = sha256.Sum256(data)
		if seen[h] {
			return nil
		}
		seen[h] = true
		files = append(files, p)
		if len(files) >= maxCorpusFiles {
			return errStop
		}
		return nil
	})
	if walkErr != nil && walkErr != errStop {
		return "", 0, 0, walkErr
	}
	if len(files) == 0 {
		return "", 0, 0, errors.New("no <lang> files in stack")
	}
	// 1MB total cap: keep the biggest files, in descending size order.
	var total int = 0
	var p string

	for _, p = range files {
		var st, err = os.Stat(p)
		if err == nil {
			total += int(st.Size())
		}
	}
	if total > maxCorpusBytes {
		sort.Slice(files, func(i int, j int) bool {
			var si fs.FileInfo

			si, _ = os.Stat(files[i])
			var sj fs.FileInfo

			sj, _ = os.Stat(files[j])
			return si.Size() > sj.Size()
		})
		var keep []string = make([]string, 0, len(files))
		var acc int = 0
		var p string

		for _, p = range files {
			var st fs.FileInfo

			st, _ = os.Stat(p)
			if acc+int(st.Size()) > maxCorpusBytes {
				continue
			}
			keep = append(keep, p)
			acc += int(st.Size())
		}
		files = keep
	}
	var buf bytes.Buffer
	{
		var p string

		for _, p = range files {
			var data, err = os.ReadFile(p)
			if err != nil {
				continue
			}
			buf.WriteString("# ==== " + filepath.Base(p) + " ====\n")
			buf.Write(bytes.ToValidUTF8(data, []byte("\uFFFD")))
			buf.WriteString("\n")
		}
	}
	if buf.Len() == 0 {
		return "", 0, 0, errors.New("corpus empty after dedupe")
	}
	{
		var err = os.WriteFile(out, buf.Bytes(), 0644)
		if err != nil {
			return "", 0, 0, err
		}
	}
	return out, len(files), buf.Len(), nil
}

// recordStackManifest: the per-stack SLM collection - stacks/<name>.json
// records every language SLM built for that stack element (each stack
// collects its own slices; the global index.json stays the router's view).
// The entry carries the full launch recipe omp-potato needs to start the
// SLM: base model, adapter, context, threads, attention window.
func recordStackManifest(lang string, stack string, ok bool) {
	var name string = filepath.Base(strings.TrimRight(stack, "/"))
	name = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '-' || r == '_' {
			return r
		}
		return '_'
	}, name)
	var dir string = fleetDir + "/stacks"
	var err = os.MkdirAll(dir, 0755)
	if err != nil {
		return
	}
	var path string = dir + "/" + name + ".json"
	var st struct {
		Stack     string                     `json:"stack"`
		Languages map[string]*stackLangEntry `json:"languages"`
	}
	st.Stack = stack
	st.Languages = make(map[string]*stackLangEntry)
	{
		var data, err = os.ReadFile(path)
		if err == nil {
			_ = json.Unmarshal(data, &st)
		}
	}
	var e stackLangEntry
	var prev, has = st.Languages[lang]
	if has {
		e = *prev
	}
	e.Status = "ready"
	if !ok {
		e.Status = "error"
	}
	e.Lora = "adapters/" + lang + ".gguf"
	e.Base = "Qwen3-0.6B-Q8_0.gguf"
	e.Ctx = 4096
	e.Threads = 4
	e.Window = 128
	e.BuiltAt = float64(time.Now().Unix())
	st.Languages[lang] = &e
	var data []byte

	data, _ = json.MarshalIndent(st, "", " ")
	var tmp string = path + ".tmp"
	if os.WriteFile(tmp, data, 0644) == nil {
		_ = os.Rename(tmp, path)
	}
}

type stackLangEntry struct {
	Status  string  `json:"status"`
	Lora    string  `json:"lora"`
	Base    string  `json:"base"`
	Ctx     int     `json:"ctx"`
	Threads int     `json:"threads"`
	Window  int     `json:"window"`
	BuiltAt float64 `json:"built_at"`
}

// stacksCmd: expertd stacks [stack] - list every stack element and its
// SLMs. With a stack argument, backfill the manifest from index.json
// (collect the SLMs that already exist for this stack's languages without
// rebuilding).
func stacksCmd(args []string) {
	var stack string = ""
	var a string

	for _, a = range args {
		stack = a
	}
	if stack != "" {
		var sc scanResult

		for _, sc = range scanDir(stack) {
			var idx = loadIndex()
			var e, ok = idx[sc.Lang]
			if ok && e.Status == "ready" {
				recordStackManifest(sc.Lang, stack, true)
			}
		}
	}
	var dir string = fleetDir + "/stacks"
	var entries, err = os.ReadDir(dir)
	if err != nil {
		fmt.Println("[stacks] no per-stack manifests yet (run the watcher on a stack)")
		return
	}
	var de fs.DirEntry

	for _, de = range entries {
		if de.IsDir() || filepath.Ext(de.Name()) != ".json" {
			continue
		}
		var data, err = os.ReadFile(filepath.Join(dir, de.Name()))
		if err != nil {
			continue
		}
		var st struct {
			Stack     string                     `json:"stack"`
			Languages map[string]*stackLangEntry `json:"languages"`
		}
		if json.Unmarshal(data, &st) != nil {
			continue
		}
		fmt.Printf("%s:\n", st.Stack)
		var lang string

		var e *stackLangEntry

		for lang, e = range st.Languages {
			fmt.Printf("  %-12s %-7s base=%s lora=%s ctx=%d t=%d w=%d\n",
				lang, e.Status, e.Base, e.Lora, e.Ctx, e.Threads, e.Window)
		}
	}
}
func publishBuild(lang string, ok bool, lora string, dt float64, errMsg string) {
	var idx = loadIndex()
	var e expertEntry
	var has bool
	e, has = idx[lang]
	if !has {
		e = expertEntry{}
	}
	if ok {
		e.Status = "ready"
		e.Lora = lora
		e.Base = "Qwen3-0.6B-Q8_0.gguf"
		e.TrainedAt = float64(time.Now().Unix())
		e.TrainSeconds = dt
		e.Error = ""
		e.ErrorAt = 0
	} else {
		e.Status = "error"
		e.Error = errMsg
		e.ErrorAt = float64(time.Now().Unix())
	}
	idx[lang] = e
	var data []byte
	data, _ = json.MarshalIndent(idx, "", " ")
	var tmp string = indexPath + ".tmp"
	if os.WriteFile(tmp, data, 0644) == nil {
		_ = os.Rename(tmp, indexPath)
	}
}
