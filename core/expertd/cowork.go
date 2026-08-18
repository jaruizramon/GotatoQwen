// cowork.go - the tool-use loop: the SLM works on the project, not just chat.
//
// ReAct-style: the model may emit EXACTLY one of
//   <tool_call>{"name":"list_dir","arguments":{"path":"..."}}</tool_call>
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

var toolCallRe = regexp.MustCompile(`<tool_call>(.*?)</tool_call>`)

func coworkPrompt(tools []mcpTool) string {
	var sb strings.Builder
	sb.WriteString("You are working inside a project. You have these tools:\n")
	for _, t := range tools {
		sb.WriteString("- " + t.Name + ": " + t.Description + "\n")
	}
	sb.WriteString("\nWhen you need to inspect or act on the project, emit EXACTLY one " +
		"<tool_call> block, for example:\n" +
		"<tool_call>{\"name\":\"list_dir\",\"arguments\":{\"path\":\"/home/pipo/stack\"}}</tool_call>\n" +
		"Use only the tool names listed above - never any other name. Then wait for " +
		"the result. When you have enough information, answer the user directly in " +
		"your final message. Never invent file contents; only report what the tools returned.")
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
	messages := []map[string]string{
		{"role": "system", "content": coworkPrompt(tools)},
		{"role": "user", "content": prompt},
	}
	for iter := 0; iter < 5; iter++ {
		body, _ := json.Marshal(map[string]any{
			"messages": messages, "temperature": 0, "max_tokens": 400,
			"enable_thinking": false})
		resp, err := httpPostJSONSlow(backend+"/v1/chat/completions", body)
		if err != nil {
			return "cowork backend unreachable: " + err.Error(), transcript.String()
		}
		var out struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if json.Unmarshal(resp, &out) != nil || len(out.Choices) == 0 {
			return "bad backend response", transcript.String()
		}
		content := strings.TrimSpace(out.Choices[0].Message.Content)
		m := toolCallRe.FindStringSubmatch(content)
		if m == nil && strings.Contains(content, "<tool_call>") {
			// tolerate an UNCLOSED call (token-budget truncation): grab from
			// the opening tag to the end and let the JSON parser decide
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
			return content, transcript.String() // final answer
		}
		var call struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if json.Unmarshal([]byte(m[1]), &call) != nil || call.Name == "" {
			// malformed call: tell the model and let it retry
			messages = append(messages,
				map[string]string{"role": "assistant", "content": content})
			var hint string = "Malformed tool call. Use the exact format."
			if call.Name != "" {
				hint = "Unknown tool '" + call.Name + "'. Available tools: "
				for _, t := range tools {
					hint += t.Name + ", "
				}
				hint += ". Emit the exact JSON."
			}
			messages = append(messages,
				map[string]string{"role": "user", "content": hint})
			continue
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
		if call.Name == "run_command" && !isErr && strings.Contains(result, "requires user approval") {
			return "run_command requires approval - skipped.", transcript.String()
		}
		transcript.WriteString(fmt.Sprintf("  [tool] %s %v -> %d chars%s\n",
			call.Name, call.Arguments, len(result), map[bool]string{true: " (error)", false: ""}[isErr]))
		messages = append(messages,
			map[string]string{"role": "assistant", "content": content})
		messages = append(messages,
			map[string]string{"role": "user", "content": "Tool result: " + result})
	}
	return "tool loop exceeded 5 iterations", transcript.String()
}

// approveCommand: y/N gate for run_command. The TUI asks interactively;
// the headless gateway has no tty, so it denies unless GOTATO_APPROVE=1.
func approveCommand(name string) bool {
	if name != "run_command" {
		return true
	}
	if os.Getenv("GOTATO_APPROVE") == "1" {
		return true
	}
	fmt.Print("  [approval] run this command? [y/N] ")
	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadString('\n')
	if err != nil {
		return false
	}
	return strings.TrimSpace(strings.ToLower(line)) == "y"
}

// chatToolBlock: the compact tool contract injected into chat prompts so
// the routed SLM can ask for tools; the gateway executes and loops with
// the instruct brain (see streamRelayChat).
func chatToolBlock() string {
	return "\nAVAILABLE TOOLS:\n" +
		"- list_dir(path): list files in a directory\n" +
		"- read_file(path): read a file\n" +
		"- run_command(command): run a shell command (approval may be required)\n" +
		"When you need to inspect files or run commands, emit EXACTLY one " +
		"<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call> and wait " +
		"for the result. Never invent file contents; only report what the tools " +
		"returned.\n"
}
