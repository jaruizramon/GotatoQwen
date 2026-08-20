// mcp.go - Model Context Protocol: a minimal stdlib client AND server.
//
// Server: spawned as a subprocess (stdio transport); speaks JSON-RPC 2.0:
//
//	initialize -> {protocolVersion, capabilities, serverInfo}
//	notifications/initialized (no id, no reply)
//	tools/list  -> the tool registry
//	tools/call  -> executes and returns {content:[{type:"text",text}], isError}
//
// Client: spawns the server, does the handshake, exposes list/call.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

type mcpTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type rpcMsg struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
}

type rpcResult struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id"`
	Result  any    `json:"result"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type toolCallResult struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	IsError bool `json:"isError"`
}

// ---- the tool registry (shared by builtin execution and the MCP server) ---
//
// Single source of truth for the fleet's tools: schemas feed the MCP
// server, the cowork tool loop, and chatToolBlock (the contract injected
// into chat prompts). SLMs are small - keep descriptions terse and
// arguments flat strings.
func toolSchemas() []mcpTool {
	return []mcpTool{
		{Name: "list_dir", Description: "List files in a directory of the project stack.",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}},
				"required":   []string{"path"}}},
		{Name: "read_file", Description: "Read a file from the project stack (optional offset/limit, 1-based lines).",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{
					"path":   map[string]any{"type": "string"},
					"offset": map[string]any{"type": "integer"},
					"limit":  map[string]any{"type": "integer"}},
				"required": []string{"path"}}},
		{Name: "edit_file", Description: "Apply exact search/replace edits to a file (approval may be required).",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{
					"path": map[string]any{"type": "string"},
					"edits": map[string]any{"type": "array",
						"items": map[string]any{"type": "object",
							"properties": map[string]any{
								"old": map[string]any{"type": "string"},
								"new": map[string]any{"type": "string"},
							},
							"required": []string{"old", "new"},
						},
					},
				},
				"required": []string{"path", "edits"},
			}},
		{Name: "write_file", Description: "Write a file inside the project stack (approval may be required).",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{
					"path":    map[string]any{"type": "string"},
					"content": map[string]any{"type": "string"}},
				"required": []string{"path", "content"}}},
		{Name: "run_command", Description: "Run a shell command in the project stack (approval may be required).",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{"command": map[string]any{"type": "string"}},
				"required":   []string{"command"}}},
		{Name: "grep_files", Description: "Regex-search files in the stack (optionally narrowed by glob).",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{
					"pattern": map[string]any{"type": "string"},
					"path":    map[string]any{"type": "string"},
					"glob":    map[string]any{"type": "string"},
					"context": map[string]any{"type": "integer"},
					"max":     map[string]any{"type": "integer"}},
				"required": []string{"pattern"}}},
		{Name: "find_files", Description: "Find files by name glob (e.g. *.py) in the stack.",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"path": map[string]any{"type": "string"},
					"type": map[string]any{"type": "string"},
					"max":  map[string]any{"type": "integer"}},
				"required": []string{"name"}}},
		{Name: "git_status", Description: "Show the stack git repo status (branch + short status).",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{}}},
		{Name: "git_diff", Description: "Show the unstaged diff of the stack git repo (optionally one path).",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}}}},
		{Name: "fetch_url", Description: "Fetch an http(s) URL and return its text, capped.",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{"url": map[string]any{"type": "string"}},
				"required":   []string{"url"}}},
		{Name: "summarize_text", Description: "Compress long text with the fleet's 2B summarizer.",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{"text": map[string]any{"type": "string"}},
				"required":   []string{"text"}}},
	}
}

// resolveStackPath: the SLM may hallucinate a path (e.g. "/project/AGENT.md"
// or "/home/pipo/stack/x"). Deterministic safety net: try the path as-is;
// if it does not exist and is not already under the stack root, retry it
// joined onto the root (stripping any leading slash). Returns the original
// path when neither exists, so error messages stay truthful.
func resolveStackPath(p string) string {
	if p == "" {
		return getStackRoot()
	}
	var abs, err = filepath.Abs(p)
	if err == nil {
		p = abs // canonical key: the read guard, fidelity check and the
		// write target must all agree on the same path for one file
	}
	{
		var err error

		_, err = os.Stat(p)
		if err == nil {
			return p
		}
	}
	var root = getStackRoot()
	if strings.HasPrefix(p, root) {
		return p
	}
	var candidate = root + "/" + strings.TrimLeft(p, "/")
	{
		var err error

		_, err = os.Stat(candidate)
		if err == nil {
			return candidate
		}
	}
	return p
}

func execTool(name string, args map[string]any, approve bool) (string, bool) {
	var path string
	var v, ok = args["path"].(string)
	if ok {
		path = v
	}
	path = resolveStackPath(path)
	var cmd string
	{
		var v, ok = args["command"].(string)
		if ok {
			cmd = v
		}
	}
	switch name {
	case "list_dir":
		var entries, err = os.ReadDir(path)
		if err != nil {
			return "error: " + err.Error(), true
		}
		var out strings.Builder
		var e fs.DirEntry

		for _, e = range entries {
			var suffix = ""
			if e.IsDir() {
				suffix = "/"
			}
			out.WriteString("  " + e.Name() + suffix + "\n")
		}
		return out.String(), false
	case "read_file":
		// stat + bounded read BEFORE anything: a huge file (e.g. a .gguf in
		// the stack) must never be loaded into RAM just to be truncated.
		var st, err = os.Stat(path)
		if err != nil {
			return "error: " + err.Error(), true
		}
		var size int64 = st.Size()
		if size > 20000 {
			size = 20000
		}
		var f *os.File

		f, err = os.Open(path)
		if err != nil {
			return "error: " + err.Error(), true
		}
		var data []byte = make([]byte, size)
		var n int

		n, err = io.ReadFull(f, data)
		_ = f.Close()
		if err != nil && err != io.ErrUnexpectedEOF {
			return "error: " + err.Error(), true
		}
		var text string = string(data[:n])
		var offset int = 0
		var limit int = 0
		var v, ok = args["offset"].(float64)
		if ok {
			offset = int(v)
		}
		{
			var v, ok = args["limit"].(float64)
			if ok {
				limit = int(v)
			}
		}
		if offset > 0 || limit > 0 {
			var lines []string = strings.Split(text, "\n")
			var start int = 0
			if offset > 0 {
				start = offset - 1
			}
			if start > len(lines) {
				start = len(lines)
			}
			var end int = len(lines)
			if limit > 0 && start+limit < end {
				end = start + limit
			}
			text = strings.Join(lines[start:end], "\n")
		}
		return text, false
	case "edit_file":
		if !approve {
			return "error: edit_file requires user approval", true
		}
		var data, err = os.ReadFile(path)
		if err != nil {
			return "error: " + err.Error(), true
		}
		var content string = string(data)
		var edits []any
		var ok bool
		if edits, ok = args["edits"].([]any); !ok {
			return "error: edit_file needs an edits array", true
		}
		var applied int = 0
		var logOut strings.Builder
		var raw any

		for _, raw = range edits {
			var em map[string]any
			var ok2 bool
			if em, ok2 = raw.(map[string]any); !ok2 {
				continue
			}
			var oldS string

			oldS, _ = em["old"].(string)
			var newS string

			newS, _ = em["new"].(string)
			if oldS == "" {
				return "error: edit_file 'old' must be non-empty", true
			}
			var n int = strings.Count(content, oldS)
			if n == 0 {
				return "error: edit_file pattern not found: " + clipToolText(oldS), true
			}
			if n > 1 {
				return fmt.Sprintf("error: edit_file pattern is ambiguous (%d matches): %s", n, clipToolText(oldS)), true
			}
			content = strings.Replace(content, oldS, newS, 1)
			applied++
			logOut.WriteString(fmt.Sprintf("  edit %d: %d chars -> %d chars\n", applied, len(oldS), len(newS)))
		}
		if applied == 0 {
			return "error: edit_file found no valid edits", true
		}
		{
			var err = os.WriteFile(path, []byte(content), 0644)
			if err != nil {
				return "error: " + err.Error(), true
			}
		}
		return fmt.Sprintf("edited %s (%d edit(s)):\n%s", path, applied, logOut.String()), false
	case "grep_files":
		var pattern string
		var v, ok = args["pattern"].(string)
		if ok {
			pattern = v
		}
		if pattern == "" {
			return "error: grep_files needs a pattern", true
		}
		var re, err = regexp.Compile(pattern)
		if err != nil {
			return "error: bad regex: " + err.Error(), true
		}
		var searchRoot string = getStackRoot()
		{
			var v, ok = args["path"].(string)
			if ok && v != "" {
				searchRoot = resolveStackPath(v)
			}
		}
		var glob string = ""
		{
			var v, ok = args["glob"].(string)
			if ok {
				glob = v
			}
		}
		var max int = 50
		{
			var v, ok = args["max"].(float64)
			if ok && v > 0 {
				max = int(v)
			}
		}
		var ctxLines int = 0
		{
			var v, ok = args["context"].(float64)
			if ok && v > 0 {
				ctxLines = int(v)
			}
		}
		var out strings.Builder
		var hits int = 0
		filepath.WalkDir(searchRoot, func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if skipSearchDir(d.Name()) {
					return filepath.SkipDir
				}
				return nil
			}
			if glob != "" {
				var ok bool

				ok, _ = filepath.Match(glob, d.Name())
				if !ok {
					return nil
				}
			}
			if hits >= max {
				return filepath.SkipAll
			}
			{
				// stat BEFORE read: the stack may hold multi-GB model files;
				// loading one to check its size is how the gateway OOMs.
				var info, err = d.Info()
				if err == nil && info.Size() > 512*1024 {
					return nil
				}
			}
			var data []byte

			data, err = os.ReadFile(p)
			if err != nil {
				return nil // unreadable or binary-ish: skip
			}
			var lines []string = strings.Split(string(data), "\n")
			var i int

			var ln string

			for i, ln = range lines {
				if !re.MatchString(ln) {
					continue
				}
				if hits >= max {
					return filepath.SkipAll
				}
				if ctxLines > 0 {
					var lo int = i - ctxLines
					if lo < 0 {
						lo = 0
					}
					var hi int = i + ctxLines
					if hi >= len(lines) {
						hi = len(lines) - 1
					}
					out.WriteString(fmt.Sprintf("-- %s:%d\n", p, i+1))
					var j = lo
					for j = lo; j <= hi; j++ {
						var marker string = " "
						if j == i {
							marker = ">"
						}
						out.WriteString(fmt.Sprintf("%s %d| %s\n", marker, j+1, strings.TrimRight(lines[j], "\r")))
					}
				} else {
					out.WriteString(fmt.Sprintf("%s:%d: %s\n", p, i+1, strings.TrimRight(ln, "\r")))
				}
				hits++
				if out.Len() > 40000 {
					return filepath.SkipAll
				}
			}
			return nil
		})
		if out.Len() == 0 {
			return "no matches", false
		}
		return out.String(), false
	case "find_files":
		var nameGlob string
		var v, ok = args["name"].(string)
		if ok {
			nameGlob = v
		}
		if nameGlob == "" {
			return "error: find_files needs a name glob", true
		}
		var searchRoot string = getStackRoot()
		{
			var v, ok = args["path"].(string)
			if ok && v != "" {
				searchRoot = resolveStackPath(v)
			}
		}
		var typeFilter string = ""
		{
			var v, ok = args["type"].(string)
			if ok {
				typeFilter = v
			}
		}
		var max int = 100
		{
			var v, ok = args["max"].(float64)
			if ok && v > 0 {
				max = int(v)
			}
		}
		var out strings.Builder
		var found int = 0
		filepath.WalkDir(searchRoot, func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if skipSearchDir(d.Name()) {
					return filepath.SkipDir
				}
				if typeFilter == "dir" {
					var ok bool

					ok, _ = filepath.Match(nameGlob, d.Name())
					if ok && found < max {
						out.WriteString(p + "/\n")
						found++
					}
				}
				return nil
			}
			if typeFilter == "dir" {
				return nil
			}
			if found >= max {
				return filepath.SkipAll
			}
			var ok bool

			ok, _ = filepath.Match(nameGlob, d.Name())
			if ok {
				out.WriteString(p + "\n")
				found++
			}
			return nil
		})
		if out.Len() == 0 {
			return "no files found", false
		}
		return out.String(), false
	case "git_status":
		var root string = getStackRoot()
		var out, err = exec.Command("git", "-C", root, "status", "--short", "--branch").CombinedOutput()
		if err != nil {
			return "git error: " + string(out) + err.Error(), true
		}
		return capToolText(string(out), 8000), false
	case "git_diff":
		var root string = getStackRoot()
		var gitArgs []string = []string{"-C", root, "diff"}
		var v, ok = args["path"].(string)
		if ok && v != "" {
			gitArgs = append(gitArgs, "--", resolveStackPath(v))
		}
		var out, err = exec.Command("git", gitArgs...).CombinedOutput()
		if err != nil {
			return "git error: " + string(out) + err.Error(), true
		}
		return capToolText(string(out), 12000), false
	case "fetch_url":
		var url string
		var v, ok = args["url"].(string)
		if ok {
			url = v
		}
		if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
			return "error: only http(s) URLs are allowed", true
		}
		var req, err = http.NewRequest("GET", url, nil)
		if err != nil {
			return "error: " + err.Error(), true
		}
		req.Header.Set("User-Agent", "gotato-mcp/1.0")
		var resp *http.Response

		resp, err = httpClient.Do(req)
		if err != nil {
			return "error: " + err.Error(), true
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return fmt.Sprintf("error: HTTP %d", resp.StatusCode), true
		}
		var data []byte

		data, err = io.ReadAll(io.LimitReader(resp.Body, 20001))
		if err != nil {
			return "error: " + err.Error(), true
		}
		var text string = string(data)
		if len(text) > 20000 {
			text = text[:20000] + "\n[truncated]"
		}
		return text, false
	case "summarize_text":
		var text string
		var v, ok = args["text"].(string)
		if ok {
			text = v
		}
		if text == "" {
			return "error: summarize_text needs text", true
		}
		if len(text) > 30000 {
			text = text[:30000]
		}
		// the 0.6B summarizer slice rambles; the 2B generalist (the brain)
		// with thinking off produces tight summaries (measured).
		var backend string = "http://127.0.0.1:8082"
		{
			var v = os.Getenv("GOTATO_SUMMARIZER_URL")
			if v != "" {
				backend = v
			}
		}
		var prompt string = "Summarize the conversation below so a fresh model can continue it without losing anything. Keep all concrete requirements, code decisions, and the user's goal. Output only the summary.\n\nConversation:\n" + text + "\n\nSummary:\n"
		var body []byte

		body, _ = json.Marshal(map[string]any{
			"messages":             []map[string]string{{"role": "user", "content": prompt}},
			"max_tokens":           256,
			"temperature":          0,
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
		})
		var out, err = httpPostJSONSlow(backend+"/v1/chat/completions", body)
		if err != nil {
			return "error: summarizer backend unreachable: " + err.Error(), true
		}
		var resp struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if json.Unmarshal(out, &resp) != nil || len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
			return "error: bad summarizer response", true
		}
		return "Summary:\n" + capToolText(resp.Choices[0].Message.Content, 8000), false
	case "write_file":
		if !approve {
			return "error: write_file requires user approval", true
		}
		var content string
		var v, ok = args["content"].(string)
		if ok {
			content = v
		}
		var target string = resolveStackPath(path)
		// writes must land inside the stack root (relative or invented
		// absolute paths are re-rooted; nothing escapes the stack)
		var root = getStackRoot()
		if !strings.HasPrefix(target, root) {
			target = root + "/" + strings.TrimLeft(target, "/")
		}
		var err = os.MkdirAll(filepath.Dir(target), 0755)
		if err != nil {
			return "error: " + err.Error(), true
		}
		{
			var err = os.WriteFile(target, []byte(content), 0644)
			if err != nil {
				return "error: " + err.Error(), true
			}
		}
		return "wrote " + target + " (" + strconv.Itoa(len(content)) + " bytes)", false
	case "run_command":
		if !approve {
			return "error: command requires user approval", true
		}
		var out, err = exec.Command("bash", "-c", cmd).CombinedOutput()
		if err != nil {
			return string(out) + "\nerror: " + err.Error(), true
		}
		return string(out), false
	}
	return "error: unknown tool " + name, true
}

// skipSearchDir: dirs the file-search tools never descend into.
func skipSearchDir(name string) bool {
	switch name {
	case ".git", "node_modules", "__pycache__":
		return true
	}
	return false
}

// capToolText: hard cap on tool output (SLM windows are small).
func capToolText(s string, n int) string {
	if len(s) > n {
		return s[:n] + fmt.Sprintf("\n[truncated to %d chars]", n)
	}
	return s
}

// clipToolText: short preview of a pattern for error messages.
func clipToolText(s string) string {
	if len(s) > 60 {
		return s[:60] + "..."
	}
	return s
}

// ---- MCP server (stdio) ----------------------------------------------------
func mcpServerLoop(approve bool) {
	var sc = bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var line = sc.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		var msg rpcMsg
		if json.Unmarshal([]byte(line), &msg) != nil {
			continue
		}
		if msg.ID == 0 {
			continue // notification
		}
		var result any
		var errText string = ""
		switch msg.Method {
		case "initialize":
			result = map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "gotato-mcp", "version": "1.0.0"},
			}
		case "tools/list":
			result = map[string]any{"tools": toolSchemas()}
		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			_ = json.Unmarshal(msg.Params, &params)
			var text, isErr = execTool(params.Name, params.Arguments, approve)
			result = toolCallResult{
				Content: []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				}{{Type: "text", Text: text}},
				IsError: isErr,
			}
		default:
			errText = "method not found: " + msg.Method
		}
		var resp = rpcResult{JSONRPC: "2.0", ID: msg.ID, Result: result}
		if errText != "" {
			resp.Result = nil
			resp.Error = &struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
			}{Code: -32601, Message: errText}
		}
		var data []byte

		data, _ = json.Marshal(resp)
		fmt.Println(string(data))
	}
}

// ---- MCP client -------------------------------------------------------------
type mcpClient struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	sc     *bufio.Scanner
	nextID int
}

func mcpConnect(serverBin string, approve bool) (*mcpClient, error) {
	if serverBin == "" {
		var self, err = os.Executable()
		if err != nil {
			return nil, err
		}
		serverBin = self
	}
	var args = []string{"mcp"}
	if approve {
		args = append(args, "--approve")
	}
	var cmd = exec.Command(serverBin, args...)
	if approve {
		cmd.Env = append(os.Environ(), "GOTATO_APPROVE=1")
	}
	var stdin, err = cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	var stdout io.ReadCloser

	stdout, err = cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = os.Stderr
	{
		var err = cmd.Start()
		if err != nil {
			return nil, err
		}
	}
	var c = &mcpClient{cmd: cmd, stdin: stdin, sc: bufio.NewScanner(stdout), nextID: 1}
	c.sc.Buffer(make([]byte, 1<<20), 1<<20)
	// handshake
	var resp json.RawMessage

	resp, err = c.roundTrip("initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "gotato-tui", "version": "1.0.0"},
	})
	if err != nil {
		return nil, err
	}
	_ = resp
	// initialized notification (no id)
	var notif = rpcMsg{JSONRPC: "2.0", Method: "notifications/initialized"}
	var data []byte

	data, _ = json.Marshal(notif)
	_, _ = c.stdin.Write(append(data, '\n'))
	return c, nil
}

func (c *mcpClient) roundTrip(method string, params any) (json.RawMessage, error) {
	c.nextID++
	var msg = rpcMsg{JSONRPC: "2.0", ID: c.nextID, Method: method}
	if params != nil {
		var p []byte

		p, _ = json.Marshal(params)
		msg.Params = p
	}
	var data []byte

	data, _ = json.Marshal(msg)
	var err error

	_, err = c.stdin.Write(append(data, '\n'))
	if err != nil {
		return nil, err
	}
	for c.sc.Scan() {
		var res rpcResult
		if json.Unmarshal(c.sc.Bytes(), &res) != nil {
			continue
		}
		if res.ID != c.nextID {
			continue
		}
		if res.Error != nil {
			return nil, fmt.Errorf("rpc error: %s", res.Error.Message)
		}
		var out []byte

		out, _ = json.Marshal(res.Result)
		return out, nil
	}
	return nil, fmt.Errorf("mcp server closed")
}

func (c *mcpClient) listTools() ([]mcpTool, error) {
	var raw, err = c.roundTrip("tools/list", nil)
	if err != nil {
		return nil, err
	}
	var out struct {
		Tools []mcpTool `json:"tools"`
	}
	if json.Unmarshal(raw, &out) != nil {
		return nil, fmt.Errorf("bad tools/list")
	}
	return out.Tools, nil
}

func (c *mcpClient) callTool(name string, args map[string]any) (string, error) {
	var raw, err = c.roundTrip("tools/call", map[string]any{"name": name, "arguments": args})
	if err != nil {
		return "", err
	}
	var res toolCallResult
	if json.Unmarshal(raw, &res) != nil {
		return "", fmt.Errorf("bad tools/call")
	}
	var out strings.Builder
	var ct struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}

	for _, ct = range res.Content {
		out.WriteString(ct.Text)
	}
	return out.String(), nil
}

func (c *mcpClient) close() {
	_ = c.stdin.Close()
	_ = c.cmd.Wait()
}
