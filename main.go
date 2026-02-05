package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os" // 导入 'os' 包以读取环境变量

	"github.com/google/uuid"
)

const (
	waveAPIURL = "https://cfapi.waveterm.dev/api/waveai"
)

// OpenAIRequest 结构体，用于解析传入的请求
type OpenAIRequest struct {
	Model  string        `json:"model"`
	Input  []OpenAIInput `json:"input"`
	Stream bool          `json:"stream"`
}

type OpenAIInput struct {
	Role    string          `json:"role"`
	Content []OpenAIContent `json:"content"`
}

type OpenAIContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// WaveAIRequest 结构体，用于构建要发送到 Wave AI 的请求
type WaveAIRequest struct {
	Model     string          `json:"model"`
	Messages  []WaveAIMessage `json:"messages"`
	Stream    bool            `json:"stream"`
	SessionId string          `json:"sessionid"`
	ClientId  string          `json:"clientid"`
}

type WaveAIMessage struct {
	Role    string          `json:"role"`
	Content []WaveAIContent `json:"content"`
}

type WaveAIContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// handleHealth 是一个简单的健康检查端点
func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// handleResponses 是核心的代理处理函数
func handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	var openAIReq OpenAIRequest
	if err := json.NewDecoder(r.Body).Decode(&openAIReq); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	waveAIReq := WaveAIRequest{
		Model:    openAIReq.Model,
		Stream:   openAIReq.Stream,
		Messages: make([]WaveAIMessage, len(openAIReq.Input)),
		// 为 Wave AI 生成随机的 SessionId 和 ClientId
		SessionId: "session-" + uuid.New().String(),
		ClientId:  "client-" + uuid.New().String(),
	}

	for i, input := range openAIReq.Input {
		waveAIReq.Messages[i] = WaveAIMessage{
			Role:    input.Role,
			Content: make([]WaveAIContent, len(input.Content)),
		}
		for j, content := range input.Content {
			waveAIReq.Messages[i].Content[j] = WaveAIContent{
				Type: content.Type,
				Text: content.Text,
			}
		}
	}

	waveAIReqBody, err := json.Marshal(waveAIReq)
	if err != nil {
		http.Error(w, "Failed to create Wave AI request", http.StatusInternalServerError)
		return
	}

	proxyReq, err := http.NewRequest(http.MethodPost, waveAPIURL, bytes.NewBuffer(waveAIReqBody))
	if err != nil {
		http.Error(w, "Failed to create proxy request", http.StatusInternalServerError)
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		http.Error(w, "Failed to forward request to Wave AI", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// 将 Wave AI 的响应头复制回原始响应
	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(resp.StatusCode)

	// 将响应体以流式方式写回客户端
	io.Copy(w, resp.Body)
}

func main() {
	// 关键改动：从环境变量中读取 PORT
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080" // 如果没有设置 PORT 环境变量，则在本地测试时默认为 8080
	}
	listenAddr := ":" + port

	http.HandleFunc("/v1/responses", handleResponses)
	http.HandleFunc("/responses", handleResponses)
	http.HandleFunc("/health", handleHealth)

	log.Printf("Starting OpenAI Responses API proxy on %s", listenAddr)
	log.Printf("Proxying to: %s", waveAPIURL)
	if err := http.ListenAndServe(listenAddr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
