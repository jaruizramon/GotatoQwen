// mcp.go - Model Context Protocol: a minimal stdlib client AND server.
//
// Server: spawned as a subprocess (stdio transport); speaks JSON-RPC 2.0:
//   initialize -> {protocolVersion, capabilities, serverInfo}
//   notifications/initialized (no id, no reply)
//   tools/list  -> the tool registry
//   tools/call  -> executes and returns {content:[{type:"text",text}], isError}
// Client: spawns the server, does the handshake, exposes list/call.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
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
func toolSchemas() []mcpTool {
	return []mcpTool{
		{Name: "list_dir", Description: "List files in a directory of the project stack.",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}},
				"required":    []string{"path"}}},
		{Name: "read_file", Description: "Read a file from the project stack.",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}},
				"required":    []string{"path"}}},
		{Name: "run_command", Description: "Run a shell command (user approval required).",
			InputSchema: map[string]any{"type": "object",
				"properties": map[string]any{"command": map[string]any{"type": "string"}},
				"required":    []string{"command"}}},
	}
}

func execTool(name string, args map[string]any, approve bool) (string, bool) {
	var path string
	if v, ok := args["path"].(string); ok {
		path = v
	}
	var cmd string
	if v, ok := args["command"].(string); ok {
		cmd = v
	}
	switch name {
	case "list_dir":
		entries, err := os.ReadDir(path)
		if err != nil {
			return "error: " + err.Error(), true
		}
		var out strings.Builder
		for _, e := range entries {
			suffix := ""
			if e.IsDir() {
				suffix = "/"
			}
			out.WriteString("  " + e.Name() + suffix + "\n")
		}
		return out.String(), false
	case "read_file":
		data, err := os.ReadFile(path)
		if err != nil {
			return "error: " + err.Error(), true
		}
		if len(data) > 20000 {
			data = data[:20000]
		}
		return string(data), false
	case "run_command":
		if !approve {
			return "error: command requires user approval", true
		}
		out, err := exec.Command("bash", "-c", cmd).CombinedOutput()
		if err != nil {
			return string(out) + "\nerror: " + err.Error(), true
		}
		return string(out), false
	}
	return "error: unknown tool " + name, true
}

// ---- MCP server (stdio) ----------------------------------------------------
func mcpServerLoop(approve bool) {
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
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
			text, isErr := execTool(params.Name, params.Arguments, approve)
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
		resp := rpcResult{JSONRPC: "2.0", ID: msg.ID, Result: result}
		if errText != "" {
			resp.Result = nil
			resp.Error = &struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
			}{Code: -32601, Message: errText}
		}
		data, _ := json.Marshal(resp)
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
		self, err := os.Executable()
		if err != nil {
			return nil, err
		}
		serverBin = self
	}
	args := []string{"mcp"}
	if approve {
		args = append(args, "--approve")
	}
	cmd := exec.Command(serverBin, args...)
	if approve {
		cmd.Env = append(os.Environ(), "GOTATO_APPROVE=1")
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	c := &mcpClient{cmd: cmd, stdin: stdin, sc: bufio.NewScanner(stdout), nextID: 1}
	c.sc.Buffer(make([]byte, 1<<20), 1<<20)
	// handshake
	resp, err := c.roundTrip("initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "gotato-tui", "version": "1.0.0"},
	})
	if err != nil {
		return nil, err
	}
	_ = resp
	// initialized notification (no id)
	notif := rpcMsg{JSONRPC: "2.0", Method: "notifications/initialized"}
	data, _ := json.Marshal(notif)
	_, _ = c.stdin.Write(append(data, '\n'))
	return c, nil
}

func (c *mcpClient) roundTrip(method string, params any) (json.RawMessage, error) {
	c.nextID++
	msg := rpcMsg{JSONRPC: "2.0", ID: c.nextID, Method: method}
	if params != nil {
		p, _ := json.Marshal(params)
		msg.Params = p
	}
	data, _ := json.Marshal(msg)
	if _, err := c.stdin.Write(append(data, '\n')); err != nil {
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
		out, _ := json.Marshal(res.Result)
		return out, nil
	}
	return nil, fmt.Errorf("mcp server closed")
}

func (c *mcpClient) listTools() ([]mcpTool, error) {
	raw, err := c.roundTrip("tools/list", nil)
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
	raw, err := c.roundTrip("tools/call", map[string]any{"name": name, "arguments": args})
	if err != nil {
		return "", err
	}
	var res toolCallResult
	if json.Unmarshal(raw, &res) != nil {
		return "", fmt.Errorf("bad tools/call")
	}
	var out strings.Builder
	for _, ct := range res.Content {
		out.WriteString(ct.Text)
	}
	return out.String(), nil
}

func (c *mcpClient) close() {
	_ = c.stdin.Close()
	_ = c.cmd.Wait()
}
