package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"strings"
	"time"

	"github.com/google/uuid"
)

const (
	waveAPIURL   = "https://cfapi.waveterm.dev/api/waveai"
	defaultModel = "gpt-5.1"
	listenAddr   = ":8080"
)

// OpenAI Responses API Request
type ResponsesRequest struct {
	Model           string                 `json:"model"`
	Input           json.RawMessage        `json:"input"`
	Instructions    string                 `json:"instructions,omitempty"`
	MaxOutputTokens int                    `json:"max_output_tokens,omitempty"`
	Temperature     float64                `json:"temperature,omitempty"`
	TopP            float64                `json:"top_p,omitempty"`
	Stream          *bool                  `json:"stream,omitempty"`
	Tools           json.RawMessage        `json:"tools,omitempty"`
	ToolChoice      interface{}            `json:"tool_choice,omitempty"`
	Reasoning       *ReasoningConfig       `json:"reasoning,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

type InputMessage struct {
	Role    string        `json:"role"`
	Content []ContentPart `json:"content"`
}

type ContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type Tool struct {
	Type       string                 `json:"type"`
	Name       string                 `json:"name,omitempty"`
	Function   *FunctionDef           `json:"function,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type FunctionDef struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type ReasoningConfig struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

// Wave API Request
type WaveRequest struct {
	Input           json.RawMessage  `json:"input"`
	Instructions    string           `json:"instructions,omitempty"`
	MaxOutputTokens int              `json:"max_output_tokens,omitempty"`
	Model           string           `json:"model"`
	Reasoning       *ReasoningConfig `json:"reasoning,omitempty"`
	Stream          bool             `json:"stream"`
	StreamOptions   *StreamOptions   `json:"stream_options,omitempty"`
	Text            *TextConfig      `json:"text,omitempty"`
	Tools           json.RawMessage  `json:"tools,omitempty"`
	ToolChoice      interface{}      `json:"tool_choice,omitempty"`
	Temperature     float64          `json:"temperature,omitempty"`
	TopP            float64          `json:"top_p,omitempty"`
}

type StreamOptions struct {
	IncludeObfuscation bool `json:"include_obfuscation"`
}

type TextConfig struct {
	Verbosity string `json:"verbosity,omitempty"`
}

func main() {
	http.HandleFunc("/v1/responses", handleResponses)
	http.HandleFunc("/responses", handleResponses)
	http.HandleFunc("/health", handleHealth)

	log.Printf("Starting OpenAI Responses API proxy on %s", listenAddr)
	log.Printf("Proxying to: %s", waveAPIURL)
	if err := http.ListenAndServe(listenAddr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ok"}`))
}

func handleResponses(w http.ResponseWriter, r *http.Request) {
	reqID := uuid.New().String()[:8]
	log.Printf("[%s] <- %s %s from %s", reqID, r.Method, r.URL.Path, r.RemoteAddr)

	if r.Method != http.MethodPost {
		log.Printf("[%s] ERROR: Method not allowed: %s", reqID, r.Method)
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("[%s] ERROR: Failed to read request body: %v", reqID, err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	log.Printf("[%s] Request body (%d bytes): %s", reqID, len(body), truncateString(string(body), 500))

	var req ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		log.Printf("[%s] ERROR: Invalid JSON: %v", reqID, err)
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	log.Printf("[%s] Parsed request: model=%s, input_messages=%d, stream=%v, max_tokens=%d",
		reqID, req.Model, len(req.Input), req.Stream, req.MaxOutputTokens)

	// Convert to Wave API request
	waveReq := convertToWaveRequest(&req)

	// Determine if streaming
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	log.Printf("[%s] Forwarding to Wave API (stream=%v)", reqID, stream)

	if stream {
		handleStreamingResponse(w, waveReq, reqID)
	} else {
		handleNonStreamingResponse(w, waveReq, reqID)
	}
}

// fixInputMessages converts "type":"message" items to regular role-based format
// This fixes the issue where Wave API requires reasoning items to be present
func fixInputMessages(input json.RawMessage) json.RawMessage {
	var items []map[string]interface{}
	if err := json.Unmarshal(input, &items); err != nil {
		return input // Return original if can't parse
	}

	var fixedItems []map[string]interface{}
	for _, item := range items {
		itemType, hasType := item["type"].(string)

		// Convert "type":"message" items to regular format
		if hasType && itemType == "message" {
			role, _ := item["role"].(string)
			content := item["content"]
			if role != "" && content != nil {
				fixedItems = append(fixedItems, map[string]interface{}{
					"role":    role,
					"content": content,
				})
				continue
			}
		}

		// Keep other items as-is
		fixedItems = append(fixedItems, item)
	}

	result, err := json.Marshal(fixedItems)
	if err != nil {
		return input
	}
	return result
}

func convertToWaveRequest(req *ResponsesRequest) *WaveRequest {
	// model := req.Model
	// if model == "" {
	// 	model = defaultModel
	// }
	model := "gpt-5.1"

	maxTokens := req.MaxOutputTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	// Fix input messages - convert "type":"message" items to regular format
	fixedInput := fixInputMessages(req.Input)

	waveReq := &WaveRequest{
		Input:           fixedInput,
		Instructions:    req.Instructions,
		MaxOutputTokens: maxTokens,
		Model:           model,
		Stream:          true,
		StreamOptions:   &StreamOptions{IncludeObfuscation: false},
		Text:            &TextConfig{Verbosity: "low"},
		Tools:           req.Tools,
		ToolChoice:      req.ToolChoice,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
	}

	if req.Reasoning != nil {
		waveReq.Reasoning = req.Reasoning
	} else {
		waveReq.Reasoning = &ReasoningConfig{Effort: "medium", Summary: "auto"}
	}

	return waveReq
}

func makeWaveRequest(waveReq *WaveRequest) (*http.Response, error) {
	body, err := json.Marshal(waveReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequest(http.MethodPost, waveAPIURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	chatID := uuid.New().String()
	clientID := uuid.New().String()

	httpReq.Header.Set("Host", "cfapi.waveterm.dev")
	httpReq.Header.Set("User-Agent", "Go-http-client/1.1")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Authorization", "Bearer")
	httpReq.Header.Set("X-Wave-Apitype", "openai-responses")
	httpReq.Header.Set("X-Wave-Chatid", chatID)
	httpReq.Header.Set("X-Wave-Clientid", clientID)
	httpReq.Header.Set("X-Wave-Requesttype", "waveai")
	httpReq.Header.Set("X-Wave-Version", "0.13.1")
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Minute}
	return client.Do(httpReq)
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func handleStreamingResponse(w http.ResponseWriter, waveReq *WaveRequest, reqID string) {
	resp, err := makeWaveRequest(waveReq)
	if err != nil {
		log.Printf("[%s] ERROR: Failed to call Wave API: %v", reqID, err)
		http.Error(w, fmt.Sprintf("Failed to call Wave API: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	log.Printf("[%s] Wave API response: %d %s", reqID, resp.StatusCode, resp.Status)

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("[%s] ERROR: Wave API error response: %s", reqID, truncateString(string(body), 500))
		http.Error(w, fmt.Sprintf("Wave API error: %s", string(body)), resp.StatusCode)
		return
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Printf("[%s] ERROR: Streaming not supported", reqID)
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Stream the response directly
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	lineCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		lineCount++
		if line == "" {
			w.Write([]byte("\n"))
			flusher.Flush()
			continue
		}
		w.Write([]byte(line + "\n"))
		flusher.Flush()
	}

	if err := scanner.Err(); err != nil {
		log.Printf("[%s] ERROR: Scanner error: %v", reqID, err)
	}

	log.Printf("[%s] -> Streaming complete, sent %d lines", reqID, lineCount)
}

func handleNonStreamingResponse(w http.ResponseWriter, waveReq *WaveRequest, reqID string) {
	resp, err := makeWaveRequest(waveReq)
	if err != nil {
		log.Printf("[%s] ERROR: Failed to call Wave API: %v", reqID, err)
		http.Error(w, fmt.Sprintf("Failed to call Wave API: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	log.Printf("[%s] Wave API response: %d %s", reqID, resp.StatusCode, resp.Status)

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("[%s] ERROR: Wave API error response: %s", reqID, truncateString(string(body), 500))
		http.Error(w, fmt.Sprintf("Wave API error: %s", string(body)), resp.StatusCode)
		return
	}

	// Collect streaming response and build final response
	var finalResponse map[string]interface{}
	var outputText strings.Builder

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	eventCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		eventCount++
		data := strings.TrimPrefix(line, "data: ")
		var event map[string]interface{}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			log.Printf("[%s] WARN: Failed to parse event: %v", reqID, err)
			continue
		}

		eventType, _ := event["type"].(string)

		switch eventType {
		case "response.output_text.delta":
			if delta, ok := event["delta"].(string); ok {
				outputText.WriteString(delta)
			}
		case "response.completed":
			if respData, ok := event["response"].(map[string]interface{}); ok {
				finalResponse = respData
			}
		case "error":
			log.Printf("[%s] ERROR: Received error event: %v", reqID, event)
		}
	}

	log.Printf("[%s] Processed %d events, output length: %d", reqID, eventCount, outputText.Len())

	if finalResponse == nil {
		log.Printf("[%s] WARN: No response.completed event, building fallback response", reqID)
		finalResponse = map[string]interface{}{
			"id":         fmt.Sprintf("resp_%s", uuid.New().String()[:32]),
			"object":     "response",
			"created_at": time.Now().Unix(),
			"status":     "completed",
			"model":      waveReq.Model,
			"output": []map[string]interface{}{
				{
					"type": "message",
					"role": "assistant",
					"content": []map[string]interface{}{
						{
							"type": "output_text",
							"text": outputText.String(),
						},
					},
				},
			},
		}
	}

	w.Header().Set("Content-Type", "application/json")
	respBytes, _ := json.Marshal(finalResponse)
	log.Printf("[%s] -> Response (%d bytes): %s", reqID, len(respBytes), truncateString(string(respBytes), 300))
	json.NewEncoder(w).Encode(finalResponse)
}
