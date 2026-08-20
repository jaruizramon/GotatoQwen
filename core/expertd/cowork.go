// cowork.go - the tool-use loop: the SLM works on the project, not just chat.
//
// ReAct-style: the model may emit EXACTLY one of
//
//	<tool_call>{"name":"list_dir","arguments":{"path":"..."}}</tool_call>
//
// The harness executes it (builtin or via MCP), feeds the result back, and
// loops until the model answers directly. run_command requires approval.
// The cowork model is the instruct 1.7B (:8086) - base models narrate
// instead of calling tools.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

var toolCallRe = regexp.MustCompile(`(?s)<tool_call>(.*?)</tool_call>`)

// extractToolCall: find a tool call in a model reply - wrapped
// (<tool_call>JSON</tool_call>), unclosed (budget truncation), or BARE
// JSON (the tiny slices drop the wrapper, often with stray whitespace
// like {"name" : "list_dir"}). Returns the call JSON. Shared by the
// cowork loop and the chat handler's tool-loop trigger so both paths
// agree on what counts as a call.
func extractToolCall(content string, tools []mcpTool) (string, bool) {
	content = strings.TrimSpace(content)
	var m = toolCallRe.FindStringSubmatch(content)
	if m != nil {
		return m[1], true
	}
	if strings.Contains(content, "<tool_call>") {
		var i = strings.Index(content, "<tool_call>")
		var tail = content[i+len("<tool_call>"):]
		var j = strings.Index(tail, "</tool_call>")
		if j >= 0 {
			tail = tail[:j]
		}
		if json.Valid([]byte(strings.TrimSpace(tail))) {
			return strings.TrimSpace(tail), true
		}
	}
	// bare JSON: accept the first balanced object naming a known tool.
	var i = 0
	for i = 0; i < len(content); {
		var j int = strings.Index(content[i:], "{")
		if j < 0 {
			break
		}
		var objStart int = i + j
		var end int = braceClose(content, objStart)
		if end < 0 {
			break
		}
		var candidate string = content[objStart:end]
		var c struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if json.Unmarshal([]byte(candidate), &c) == nil && c.Name != "" {
			var known bool = false
			var t mcpTool

			for _, t = range tools {
				if t.Name == c.Name {
					known = true
					break
				}
			}
			if known {
				return candidate, true
			}
		}
		i = objStart + 1
	}
	return "", false
}

// verifyURL: the verifier SLM (the 2B generalist) that checks a peer SLM's
// writes. Set by the gateway (serveCmd). The fragmented-LLM contract: one
// SLM executes, a second SLM verifies - correctness is worth the extra
// seconds of delegation.
var verifyURL string = ""

// stackRoot: the gateway's cwd is the project stack the tools operate on
// (execTool resolves relative paths against it). Spelled out in the tool
// prompts so the SLM stops inventing plausible paths like "app/".
var stackRoot string

func getStackRoot() string {
	if stackRoot == "" {
		var wd, err = os.Getwd()
		if err == nil {
			stackRoot = wd
		}
	}
	return stackRoot
}

func coworkPrompt(tools []mcpTool) string {
	var sb strings.Builder
	sb.WriteString("You are working inside a project. The project stack root is \"" +
		getStackRoot() + "\". All file paths are ABSOLUTE paths under this " +
		"root - never invent paths. You have these tools:\n")
	var t mcpTool

	for _, t = range tools {
		sb.WriteString("- " + t.Name + ": " + t.Description + "\n")
	}
	sb.WriteString("\nWhen you need to inspect or act on the project, emit EXACTLY one " +
		"<tool_call> block, for example:\n")
	sb.WriteString(fmt.Sprintf("<tool_call>{\"name\":\"list_dir\",\"arguments\":{\"path\":\"%s\"}}</tool_call>\n", getStackRoot()))
	sb.WriteString("Use only the tool names listed above - never any other name. Then wait for " +
		"the result. When you have enough information, answer the user directly in " +
		"your final message. Never invent file contents; only report what the tools returned. " +
		"To EDIT a file: read it first, then emit edit_file with the exact old/new text " +
		"(or write_file with the FULL modified content when rewriting most of it) - " +
		"never run_command for edits. Act now: read the file you need, then edit or write it.")
	return sb.String()
}

// coworkTurn: one user request, looped through the tool-use cycle.
// Returns the final answer and the iteration transcript.
func coworkTurn(backend string, session string, prompt string, useMCP bool) (string, string) {
	var client *mcpClient
	var err error
	if useMCP {
		client, err = mcpConnect("", false)
		if err != nil {
			return "MCP connect failed: " + err.Error(), ""
		}
		defer client.close()
	}
	var tools []mcpTool
	if useMCP {
		tools, err = client.listTools()
		if err != nil {
			return "MCP tools/list failed: " + err.Error(), ""
		}
	} else {
		tools = toolSchemas()
	}

	var transcript strings.Builder
	var wrote bool = false
	var nudge int = 0
	var verifyNudge int = 0
	var readContent map[string]string = make(map[string]string)
	var lastWritePath string = ""
	var lastWriteContent string = ""
	var messages = []map[string]string{
		{"role": "system", "content": coworkPrompt(tools)},
		{"role": "user", "content": prompt},
	}
	var iter = 0
	for iter = 0; iter < 4; iter++ {
		var body []byte

		body, _ = json.Marshal(map[string]any{
			"messages": messages, "temperature": 0, "max_tokens": 1500,
			"enable_thinking": false})
		var resp, err = httpPostJSONSlow(backend+"/v1/chat/completions", body)
		if err != nil {
			return "cowork backend unreachable: " + err.Error(), transcript.String()
		}
		var out struct {
			Choices []struct {
				Message struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if json.Unmarshal(resp, &out) != nil || len(out.Choices) == 0 {
			// the backend may have returned a 500 (context overflow etc.):
			// surface a hint instead of a dead error
			var errBody struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			var hint string = "bad backend response"
			if json.Unmarshal(resp, &errBody) == nil && errBody.Error.Message != "" {
				hint = "tool loop error: " + errBody.Error.Message
			}
			return hint, transcript.String()
		}
		// the Qwen3 family thinks in reasoning_content and may leave content
		// empty (thinking-mode flip-flop per slot state): the loop parses
		// tool calls from BOTH; a thinking-only turn is surfaced as-is (the
		// user wants to see the reasoning).
		var content string = out.Choices[0].Message.Content
		if content == "" {
			content = out.Choices[0].Message.ReasoningContent
		}
		content = strings.TrimSpace(content)
		var callJSON, hasCall = extractToolCall(content, tools)
		if !hasCall {
			// final-answer honesty guard: an edit task that ends without a
			// write_file is a hallucination (the 1.7B embellishes when lazy:
			// it claimed success, then claimed the file already had the
			// color). Nudge once and continue - the write must happen.
			var taskEdit bool = regexp.MustCompile(`(?i)(change|edit|update|modify|fix|set|replace|write|add|remove|delete)`).MatchString(prompt)
			var lower string = strings.ToLower(content)
			var claimsEdit bool = strings.Contains(lower, "chang") ||
				strings.Contains(lower, "edit") || strings.Contains(lower, "updat") ||
				strings.Contains(lower, "modif") || strings.Contains(lower, "wrote") ||
				strings.Contains(lower, "already") || strings.Contains(lower, "done") ||
				strings.Contains(lower, "complete")
			if (taskEdit || claimsEdit) && !wrote && nudge < 1 {
				nudge++
				messages = append(messages,
					map[string]string{"role": "user",
						"content": "No write_file was executed. Emit EXACTLY one write_file tool call now: path = the file you read, content = its EXACT content from the tool result with ONLY the requested change applied. Then wait."})
				continue
			}
			// ---- verifier delegation: a second SLM (the 2B) checks the
			// executor's write before the answer ships. Correctness over
			// speed: the extra ~10s is the price of a correct edit.
			if wrote && lastWritePath != "" && verifyURL != "" {
				var orig, ok = readContent[lastWritePath]
				if ok &&
					len(orig) <= 3000 && len(lastWriteContent) <= 3000 && verifyNudge < 1 {
					var okV bool = false
					var reason string = ""
					okV, reason = verifyEdit(prompt, orig, lastWriteContent)
					if okV {
						transcript.WriteString("  [verify] 2B: VERIFIED\n")
						fmt.Fprintln(os.Stderr, "[verify] 2B: VERIFIED")
					} else {
						verifyNudge++
						transcript.WriteString("  [verify] 2B: NO - " + reason + "\n")
						fmt.Fprintf(os.Stderr, "[verify] 2B: NO - %s\n", reason)
						messages = append(messages,
							map[string]string{"role": "user",
								"content": "A second model verified your write and rejected it: " + reason + ". Re-emit write_file fixing exactly that."})
						continue
					}
				}
			}
			return content, transcript.String() // final answer
		}
		var call struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if json.Unmarshal([]byte(callJSON), &call) != nil || call.Name == "" {
			// malformed call: tell the model and let it retry
			messages = append(messages,
				map[string]string{"role": "assistant", "content": content})
			var hint string = "Malformed tool call. Use the exact format."
			if call.Name != "" {
				hint = "Unknown tool '" + call.Name + "'. Available tools: "
				var t mcpTool

				for _, t = range tools {
					hint += t.Name + ", "
				}
				hint += ". Emit the exact JSON."
			}
			messages = append(messages,
				map[string]string{"role": "user", "content": hint})
			continue
		}
		// write pre-check: a fragment write must NEVER land on disk - the
		// fidelity check runs BEFORE execTool (a rejected write must mean
		// untouched; the model may never repair a clobbered file).
		if call.Name == "write_file" {
			var p string = ""
			var c string = ""
			var v, ok = call.Arguments["path"].(string)
			if ok {
				p = resolveStackPath(v)
			}
			{
				var v, ok = call.Arguments["content"].(string)
				if ok {
					c = v
				}
			}
			{
				var prev, ok = readContent[p]
				if ok && len(prev) > 0 && len(c) < len(prev)/2 {
					transcript.WriteString(fmt.Sprintf("  [reject] write_file %s: %d chars vs %d read (fragment - untouched)\n", p, len(c), len(prev)))
					messages = append(messages,
						map[string]string{"role": "user",
							"content": fmt.Sprintf("Your write_file was REJECTED before writing: %d chars but the file you read has %d. Re-emit write_file with the COMPLETE content from the read result, changing ONLY the requested part.", len(c), len(prev))})
					continue
				}
			}
		}
		// read pre-check: a file already read in this loop must not be
		// re-executed (the model spins on re-reads instead of writing).
		if call.Name == "read_file" {
			var p string = ""
			var v, ok = call.Arguments["path"].(string)
			if ok {
				p = resolveStackPath(v)
			}
			{
				var ok bool

				_, ok = readContent[p]
				if ok {
					messages = append(messages,
						map[string]string{"role": "user",
							"content": "You already read this file - its content is above in the tool result. Do NOT read it again. Emit write_file with the FULL modified content, or answer the user."})
					continue
				}
			}
		}
		var result string
		var isErr bool
		if useMCP {
			result, err = client.callTool(call.Name, call.Arguments)
			if err != nil {
				result = "error: " + err.Error()
			}
		} else {
			result, isErr = execTool(call.Name, call.Arguments, approveCommand(call.Name))
		}
		// the brain's window is 4096: a big read (or a long file) must not
		// overflow the loop context - truncate what comes back.
		if len(result) > 4000 {
			result = result[:4000] + "\n[truncated to 4000 chars]"
		}
		if call.Name == "run_command" && !isErr && strings.Contains(result, "requires user approval") {
			return "run_command requires approval - skipped.", transcript.String()
		}
		transcript.WriteString(fmt.Sprintf("  [tool] %s %v -> %d chars%s\n",
			call.Name, call.Arguments, len(result), map[bool]string{true: " (error)", false: ""}[isErr]))
		fmt.Fprintf(os.Stderr, "[tool] %s %v -> %d chars%s\n",
			call.Name, call.Arguments, len(result), map[bool]string{true: " (error)", false: ""}[isErr])
		if call.Name == "read_file" && !isErr {
			var p string = ""
			var v, ok = call.Arguments["path"].(string)
			if ok {
				p = resolveStackPath(v) // canonical key: the guard and the
				// fidelity check must see the same path the write targets
			}
			readContent[p] = result
		}
		if call.Name == "write_file" && !isErr {
			wrote = true
			// track the accepted write for the verifier delegation (the
			// length rejection already happened pre-write, above)
			var p string = ""
			var c string = ""
			var v, ok = call.Arguments["path"].(string)
			if ok {
				p = resolveStackPath(v)
			}
			{
				var v, ok = call.Arguments["content"].(string)
				if ok {
					c = v
				}
			}
			lastWritePath = p
			lastWriteContent = c
		}
		messages = append(messages,
			map[string]string{"role": "assistant", "content": content})
		messages = append(messages,
			map[string]string{"role": "user", "content": "Tool result: " + result})
	}
	return "tool loop exceeded 7 iterations", transcript.String()
}

// verifyEdit: delegate the write's correctness to a SECOND SLM (the 2B
// generalist - bigger, better at comparing). Returns (verified, reason).
// An ambiguous verdict never blocks the executor (trust on doubt).
func verifyEdit(task string, orig string, written string) (bool, string) {
	var body []byte

	body, _ = json.Marshal(map[string]any{
		"messages": []map[string]string{{
			"role": "user",
			"content": fmt.Sprintf("Task: %s\n\nORIGINAL file (%d chars):\n%s\n\nWRITTEN file (%d chars):\n%s\n\nDid the write apply the requested change and preserve everything else? Reply VERIFIED or NO <one-line reason>.",
				task, len(orig), orig, len(written), written),
		}},
		"max_tokens": 40, "temperature": 0,
		"chat_template_kwargs": map[string]any{"enable_thinking": false}})
	var resp, err = httpPostJSONSlow(verifyURL+"/v1/chat/completions", body)
	if err != nil {
		return true, ""
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(resp, &out) != nil || len(out.Choices) == 0 {
		return true, ""
	}
	var verdict string = strings.ToUpper(strings.TrimSpace(out.Choices[0].Message.Content))
	if strings.HasPrefix(verdict, "VERIFIED") {
		return true, ""
	}
	if strings.HasPrefix(verdict, "NO") {
		return false, strings.TrimSpace(out.Choices[0].Message.Content)
	}
	return true, ""
}

// approveCommand: y/N gate for run_command. The TUI asks interactively;
// the headless gateway has no tty, so it denies unless GOTATO_APPROVE=1.
func approveCommand(name string) bool {
	if name != "run_command" && name != "write_file" && name != "edit_file" {
		return true
	}
	if os.Getenv("GOTATO_APPROVE") == "1" {
		return true
	}
	fmt.Printf("  [approval] %s? [y/N] ", name)
	var reader = bufio.NewReader(os.Stdin)
	var line, err = reader.ReadString('\n')
	if err != nil {
		return false
	}
	return strings.TrimSpace(strings.ToLower(line)) == "y"
}

// chatToolBlock: the compact tool contract injected into chat prompts so
// the routed SLM can ask for tools; the gateway executes and loops with
// the instruct brain (see streamRelayChat).
func chatToolBlock() string {
	var sb strings.Builder
	sb.WriteString("\nAVAILABLE TOOLS:\n")
	var t mcpTool

	for _, t = range toolSchemas() {
		sb.WriteString("- " + t.Name + ": " + t.Description + "\n")
	}
	sb.WriteString("The project stack root is \"" + getStackRoot() + "\". Use ABSOLUTE paths " +
		"under it when calling tools - never invented paths. " +
		"Tools are OPTIONAL: if the answer does not need the filesystem, answer " +
		"directly in this turn without any tool call. When the task DOES need " +
		"inspection or edits, do not narrate your plan - emit exactly one " +
		"<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call> and wait " +
		"for the result, then finish. Never invent file contents; only report what " +
		"the tools returned.\n")
	return sb.String()
}
