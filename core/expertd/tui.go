// tui.go - the full-screen harness TUI, blit-per-line like Claude Code /
// Codex / OMP / Pi: alternate screen, block transcript, live status line,
// bottom input. Raw mode via Linux termios (stdlib syscall only).
//
//   chat mode   : SSE-streamed answers with a live Thinking [SLM-tag] line
//   cowork mode : tool calls rendered as blocks as they execute
//
// Keys: Enter submit, Backspace edit, Ctrl-U clear line, Ctrl-C interrupt or
// quit at the prompt, Up/Down history, Ctrl-D quit.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// ---- termios raw mode (Linux) ---------------------------------------------
type termios struct {
	Iflag, Oflag, Cflag, Lflag uint32
	Line                       uint8
	Cc                         [32]uint8
	Ispeed, Ospeed             uint32
}

func ioctlTerm(fd uintptr, req uintptr, arg unsafe.Pointer) error {
	_, _, e := syscall.Syscall(syscall.SYS_IOCTL, fd, req, uintptr(arg))
	if e != 0 {
		return e
	}
	return nil
}

func termRaw() func() {
	var t termios
	if ioctlTerm(0, syscall.TCGETS, unsafe.Pointer(&t)) != nil {
		return func() {}
	}
	saved := t
	t.Iflag &^= syscall.IGNBRK | syscall.BRKINT | syscall.PARMRK | syscall.ISTRIP |
		syscall.INLCR | syscall.IGNCR | syscall.ICRNL | syscall.IXON
	t.Lflag &^= syscall.ECHO | syscall.ICANON | syscall.IEXTEN | syscall.ISIG
	t.Cflag &^= syscall.CSIZE | syscall.PARENB
	t.Cflag |= syscall.CS8
	t.Cc[syscall.VMIN] = 1
	t.Cc[syscall.VTIME] = 0
	_ = ioctlTerm(0, syscall.TCSETS, unsafe.Pointer(&t))
	return func() { _ = ioctlTerm(0, syscall.TCSETS, unsafe.Pointer(&saved)) }
}

func termSize() (int, int) {
	var ws struct{ Row, Col, X, Y uint16 }
	if ioctlTerm(0, 0x5413, unsafe.Pointer(&ws)) != nil { // TIOCGWINSZ
		return 24, 80
	}
	return int(ws.Row), int(ws.Col)
}

// ---- colors ----------------------------------------------------------------
const (
	cReset  = "\033[0m"
	cBold   = "\033[1m"
	cDim    = "\033[2m"
	cRev    = "\033[7m"
	cYellow = "\033[33m"
	cCyan   = "\033[36m"
	cGreen  = "\033[32m"
	cRed    = "\033[31m"
)

// ---- view model -------------------------------------------------------------
type blockKind int

const (
	bMeta blockKind = iota
	bUser
	bAssistant
	bTool
)

type block struct {
	kind blockKind
	tag  string
	text string
}

type view struct {
	mu     sync.Mutex
	blocks []block
	input  string
	status string
	width  int
}

var theView = &view{}
var outMu sync.Mutex

// slmColor: deterministic per-SLM color. Known fleet members get fixed
// colors; unknown bases (autostarted slices, new experts) get a stable
// palette color from their name hash, so the same SLM always renders the
// same color.
func slmColor(tag string) string {
	base := tag
	if i := strings.Index(tag, " \u00b7 "); i >= 0 {
		base = tag[:i]
	}
	switch base {
	case "python-expert":
		return "\033[92m" // bright green
	case "2b-general":
		return "\033[96m" // bright cyan
	case "4b-general":
		return "\033[93m" // bright yellow
	case "cowork(1.7b-instruct)", "1.7b-instruct":
		return "\033[95m" // bright magenta
	}
	var palette []string = []string{
		"\033[91m", "\033[94m", "\033[95m", "\033[93m", "\033[92m", "\033[96m"}
	var h int = 0
	var i int = 0
	for i = 0; i < len(base); i++ {
		h = (h*31 + int(base[i])) % 997
	}
	return palette[h%len(palette)]
}

func wrapText(text string, width int) string {
	if width <= 0 {
		return text
	}
	var sb strings.Builder
	var line strings.Builder
	for _, r := range text {
		if r == '\n' {
			sb.WriteString(line.String() + "\n")
			line.Reset()
			continue
		}
		line.WriteRune(r)
		if line.Len() >= width {
			sb.WriteString(line.String() + "\n")
			line.Reset()
		}
	}
	sb.WriteString(line.String())
	return sb.String()
}

func (v *view) render() {
	v.mu.Lock()
	blocks := v.blocks
	input := v.input
	status := v.status
	v.mu.Unlock()
	_, w := termSize()
	v.width = w
	var sb strings.Builder
	sb.WriteString("\033[H\033[2J")
	for _, b := range blocks {
		switch b.kind {
		case bMeta:
			if b.text != "" {
				sb.WriteString(cDim + "  " + b.text + cReset + "\n\n")
			}
		case bUser:
			sb.WriteString(cGreen + cBold + "  you> " + cReset + "\n")
			sb.WriteString(wrapText(b.text, w-4) + "\n\n")
		case bAssistant:
			if b.tag != "" {
				sb.WriteString(cDim + "  ── " + cBold + slmColor(b.tag) + b.tag +
					cReset + cDim + " ──" + cReset + "\n")
			}
			sb.WriteString(wrapText(b.text, w-4) + "\n\n")
		case bTool:
			sb.WriteString(cCyan + "  ⚙ " + b.tag + cReset + "\n")
			if b.text != "" {
				sb.WriteString(cDim + wrapText(b.text, w-8) + cReset + "\n")
			}
			sb.WriteString("\n")
		}
	}
	if status != "" {
		sb.WriteString(status + "\n\n")
	}
	sb.WriteString(cGreen + "  you> " + cReset + input + " ")
	outMu.Lock()
	_, _ = os.Stdout.WriteString(sb.String())
	outMu.Unlock()
}

func pushBlock(kind blockKind, tag string, text string) {
	theView.mu.Lock()
	theView.blocks = append(theView.blocks, block{kind: kind, tag: tag, text: text})
	theView.mu.Unlock()
}

func setStatus(s string) {
	theView.mu.Lock()
	theView.status = s
	theView.mu.Unlock()
}

func appendAssistant(tag string, text string) {
	theView.mu.Lock()
	if len(theView.blocks) == 0 || theView.blocks[len(theView.blocks)-1].kind != bAssistant {
		theView.blocks = append(theView.blocks, block{kind: bAssistant, tag: tag})
	}
	last := &theView.blocks[len(theView.blocks)-1]
	last.text += text
	theView.mu.Unlock()
}

// ---- key events --------------------------------------------------------------
type keyEvent int

const (
	keyChar keyEvent = iota
	keyEnter
	keyBackspace
	keyCtrlC
	keyCtrlD
	keyCtrlU
	keyUp
	keyDown
)

type keyMsg struct {
	kind keyEvent
	ch   byte
}

func stdinLoop(keys chan keyMsg) {
	reader := bufio.NewReader(os.Stdin)
	for {
		b, err := reader.ReadByte()
		if err != nil {
			close(keys)
			return
		}
		switch b {
		case 13, 10:
			keys <- keyMsg{kind: keyEnter}
		case 127, 8:
			keys <- keyMsg{kind: keyBackspace}
		case 3:
			keys <- keyMsg{kind: keyCtrlC}
		case 4:
			keys <- keyMsg{kind: keyCtrlD}
		case 21:
			keys <- keyMsg{kind: keyCtrlU}
		case 27:
			b2, _ := reader.ReadByte()
			if b2 == 91 {
				b3, _ := reader.ReadByte()
				if b3 == 65 {
					keys <- keyMsg{kind: keyUp}
				} else if b3 == 66 {
					keys <- keyMsg{kind: keyDown}
				}
			}
		default:
			if b >= 32 && b < 127 {
				keys <- keyMsg{kind: keyChar, ch: b}
			}
		}
	}
}

// ---- the main loop ------------------------------------------------------------
var turnDone = make(chan bool, 4)

func chatCmd(args []string) {
	var gateway string = "http://localhost:8090"
	var session string = "tui-" + fmt.Sprint(time.Now().Unix()%100000)
	var cowork bool = false
	var useMCP bool = false
	var coworkBackend string = "http://127.0.0.1:8086"
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--gateway":
			if i+1 < len(args) {
				gateway = args[i+1]
				i++
			}
		case "--session":
			if i+1 < len(args) {
				session = args[i+1]
				i++
			}
		case "--cowork":
			cowork = true
		case "--mcp":
			cowork = true
			useMCP = true
		case "--cowork-backend":
			if i+1 < len(args) {
				coworkBackend = args[i+1]
				i++
			}
		}
	}

	restore := termRaw()
	defer restore()
	fmt.Print("\033[?1049h\033[?25l")
	defer fmt.Print("\033[?25h\033[?1049l")

	mode := "chat"
	if cowork {
		mode = "cowork"
	}
	pushBlock(bMeta, "", "GotatoQwen harness · "+mode+
		" · session "+session+" · Ctrl-C quit")
	theView.render()

	keys := make(chan keyMsg, 64)
	go stdinLoop(keys)

	var history []string
	var histPos int = 0
	var input strings.Builder
	var busy bool = false
	var abort chan bool = nil
	tick := time.NewTicker(60 * time.Millisecond)
	defer tick.Stop()

	for {
		select {
		case <-tick.C:
			if busy {
				theView.render() // live spinner / elapsed
			}
		case <-turnDone:
			busy = false
			abort = nil
			theView.render()
		case k, ok := <-keys:
			if !ok {
				if busy {
					// stdin closed mid-turn: let the turn finish, then exit
					<-turnDone
				}
				return
			}
			if busy {
				if k.kind == keyCtrlC && abort != nil {
					close(abort)
					abort = nil
					setStatus(cRed + "  interrupted" + cReset)
					theView.render()
				}
				continue
			}
			switch k.kind {
			case keyCtrlC, keyCtrlD:
				return
			case keyEnter:
				line := strings.TrimSpace(input.String())
				input.Reset()
				if line == "" {
					theView.render()
					continue
				}
				if line == "/quit" || line == "/exit" {
					return
				}
				history = append(history, line)
				histPos = len(history)
				pushBlock(bUser, "", line)
				theView.render()
				busy = true
				abort = make(chan bool)
				if cowork {
					go coworkRun(coworkBackend, session, line, useMCP, abort)
				} else {
					go chatRun(gateway, session, line, abort)
				}
			case keyBackspace:
				s := input.String()
				if len(s) > 0 {
					input.Reset()
					input.WriteString(s[:len(s)-1])
				}
				theView.render()
			case keyCtrlU:
				input.Reset()
				theView.render()
			case keyUp:
				if histPos > 0 {
					histPos--
					input.Reset()
					input.WriteString(history[histPos])
					theView.render()
				}
			case keyDown:
				if histPos < len(history)-1 {
					histPos++
					input.Reset()
					input.WriteString(history[histPos])
					theView.render()
				}
			case keyChar:
				input.WriteByte(k.ch)
				theView.render()
			}
		}
	}
}

func finishTurn() {
	select {
	case turnDone <- true:
	default:
	}
}

// ---- chat mode: SSE streaming from the gateway -------------------------------
func chatRun(gateway string, session string, prompt string, abort chan bool) {
	body, _ := json.Marshal(map[string]any{
		"prompt": prompt, "n_predict": 256, "session": session, "stream": true})
	req, err := http.NewRequest("POST", gateway+"/completion", strings.NewReader(string(body)))
	if err != nil {
		pushBlock(bTool, "error", err.Error())
		finishTurn()
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		pushBlock(bTool, "error", "gateway unreachable: "+err.Error())
		finishTurn()
		return
	}
	defer resp.Body.Close()
	slm := resp.Header.Get("X-Gotato-SLM")
	if slm == "" {
		slm = resp.Header.Get("X-Gotato-Backend")
	}
	tag := slm
	start := time.Now()
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	var thinking bool = false
	var content strings.Builder
	spinner := []string{"|", "/", "-", "\\"}
	var si int = 0
	for {
		select {
		case <-abort:
			finishTurn()
			theView.render()
			return
		default:
		}
		if !sc.Scan() {
			break
		}
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		var frame struct {
			Content string `json:"content"`
			Stop    bool   `json:"stop"`
		}
		if json.Unmarshal([]byte(strings.TrimSpace(line[5:])), &frame) != nil {
			continue
		}
		if frame.Content != "" {
			content.WriteString(frame.Content)
			cur := content.String()
			if strings.Contains(cur, "<think>") && !strings.Contains(cur, "</think>") {
				thinking = true
				setStatus(cYellow + "  ⟳ Thinking... " + cBold + slmColor(tag) + tag +
					cReset + cYellow + " " + spinner[si%4] + " " +
					fmt.Sprintf("%.0fs", time.Since(start).Seconds()) + cReset)
			}
			if thinking {
				if i := strings.Index(cur, "</think>"); i >= 0 {
					thinking = false
					setStatus("")
					rest := cur[i+len("</think>"):]
					if rest != "" {
						appendAssistant(tag, rest)
					}
				}
			} else if !strings.Contains(cur, "<think>") {
				appendAssistant(tag, frame.Content)
			}
			si++
			theView.render()
		}
		if frame.Stop {
			break
		}
	}
	setStatus("")
	finishTurn()
	theView.render()
}

// ---- cowork mode: tool loop with live tool blocks ----------------------------
func coworkRun(backend string, session string, prompt string, useMCP bool, abort chan bool) {
	var client *mcpClient
	var err error
	if useMCP {
		client, err = mcpConnect("", false)
		if err != nil {
			pushBlock(bTool, "mcp", "connect failed: "+err.Error())
			finishTurn()
			return
		}
		defer client.close()
	}
	var tools []mcpTool
	if useMCP {
		tools, err = client.listTools()
		if err != nil {
			pushBlock(bTool, "mcp", err.Error())
			finishTurn()
			return
		}
	} else {
		tools = toolSchemas()
	}
	messages := []map[string]string{
		{"role": "system", "content": coworkPrompt(tools)},
		{"role": "user", "content": prompt},
	}
	for iter := 0; iter < 5; iter++ {
		select {
		case <-abort:
			finishTurn()
			theView.render()
			return
		default:
		}
		setStatus(cYellow + "  ⟳ " + strings.Repeat("·", iter+1) + cReset)
		theView.render()
		body, _ := json.Marshal(map[string]any{
			"messages": messages, "temperature": 0, "max_tokens": 400,
			"enable_thinking": false})
		resp, err := httpPostJSON(backend+"/v1/chat/completions", body)
		if err != nil {
			pushBlock(bTool, "error", "backend unreachable")
			finishTurn()
			return
		}
		var out struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if json.Unmarshal(resp, &out) != nil || len(out.Choices) == 0 {
			pushBlock(bTool, "error", "bad backend response")
			finishTurn()
			return
		}
		content := strings.TrimSpace(out.Choices[0].Message.Content)
		m := toolCallRe.FindStringSubmatch(content)
		if m == nil && strings.Contains(content, "<tool_call>") {
			i := strings.Index(content, "<tool_call>")
			tail := content[i+len("<tool_call>"):]
			if j := strings.Index(tail, "</tool_call>"); j >= 0 {
				tail = tail[:j]
			}
			if json.Valid([]byte(strings.TrimSpace(tail))) {
				m = []string{"", strings.TrimSpace(tail)}
			}
		}
		if m == nil {
			answer := strings.ReplaceAll(content, "<tool_call>", "")
			answer = strings.ReplaceAll(answer, "</tool_call>", "")
			answer = strings.TrimSpace(answer)
			if answer != "" {
				pushBlock(bAssistant, "cowork(1.7b-instruct)", answer) // tag colored in render
			}
			setStatus("")
			finishTurn()
			theView.render()
			return
		}
		var call struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if json.Unmarshal([]byte(m[1]), &call) != nil || call.Name == "" {
			messages = append(messages,
				map[string]string{"role": "assistant", "content": content})
			messages = append(messages,
				map[string]string{"role": "user", "content": "Malformed tool call. Use the exact format."})
			continue
		}
		var result string
		var isErr bool
		if useMCP {
			result, err = client.callTool(call.Name, call.Arguments)
			if err != nil {
				result = "error: " + err.Error()
				isErr = true
			}
		} else {
			result, isErr = execTool(call.Name, call.Arguments, false)
		}
		label := call.Name + " " + jsonArgsShort(call.Arguments)
		if isErr {
			pushBlock(bTool, label+" ✗", result)
		} else {
			pushBlock(bTool, label+" ✓", truncateText(result, 400))
		}
		theView.render()
		messages = append(messages,
			map[string]string{"role": "assistant", "content": content})
		messages = append(messages,
			map[string]string{"role": "user", "content": "Tool result: " + result})
	}
	pushBlock(bTool, "loop", "exceeded 5 iterations")
	setStatus("")
	finishTurn()
	theView.render()
}

func jsonArgsShort(args map[string]any) string {
	var sb strings.Builder
	for k, v := range args {
		sb.WriteString(fmt.Sprintf("%s=%v ", k, v))
	}
	return strings.TrimSpace(sb.String())
}

func truncateText(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
