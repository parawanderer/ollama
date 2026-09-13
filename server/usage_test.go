package server

import (
	"database/sql"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestUsageStoreRecordsGenerationsAndLoads(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, serverDBName)
	u, err := openServerDB(dir)
	if err != nil {
		t.Fatal(err)
	}
	cached := 4097
	numCtx := 8192
	u.recordGeneration(usageGeneration{
		At: time.Now(), Model: "m",
		Timings: api.GenerationTimings{PromptTokens: 4098, PromptTokensCached: &cached, PromptMs: 3.4, EvalMs: 111.5, Decoded: 40},
		Meta: &api.GenerationMeta{
			Hint:  &api.RequestHint{Use: "agent", Session: "wml-1", Synthetic: true},
			Shape: &api.RequestShape{Endpoint: "chat", Surface: "openai", Stream: true, Messages: 3, Tools: 2, NumCtx: &numCtx, Client: "abc"},
		},
		Devices: "CUDA0", NumCtx: 8192, NumBatch: 512, Split: "none",
	})
	u.recordLoad(usageLoad{At: time.Now(), Model: "m", Estimate: &api.LoadEstimate{Predicted: 100, Source: "probe"}, SizeVRAM: 99})
	u.close()

	db, err := sql.Open("sqlite3", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var use, session, surface string
	var tools, synthetic, reqCtx int
	if err := db.QueryRow(`SELECT hint_use, hint_session, surface, tools, hint_synthetic, req_num_ctx FROM generations`).
		Scan(&use, &session, &surface, &tools, &synthetic, &reqCtx); err != nil {
		t.Fatal(err)
	}
	if use != "agent" || session != "wml-1" || surface != "openai" || tools != 2 || synthetic != 1 || reqCtx != 8192 {
		t.Errorf("generation row: use=%q session=%q surface=%q tools=%d synthetic=%d req_num_ctx=%d", use, session, surface, tools, synthetic, reqCtx)
	}
	var source string
	var predicted, vram int64
	if err := db.QueryRow(`SELECT source, predicted, size_vram FROM loads`).Scan(&source, &predicted, &vram); err != nil {
		t.Fatal(err)
	}
	if source != "probe" || predicted != 100 || vram != 99 {
		t.Errorf("load row: source=%q predicted=%d size_vram=%d", source, predicted, vram)
	}
}

// Nothing a person wrote may reach the store: the row is built from counts and options only.
func TestChatMetaCarriesShapeNotContent(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	stream := false
	meta := chatMeta(c, api.ChatRequest{
		Messages:  []api.Message{{Role: "user", Content: "secret words", Images: []api.ImageData{{1}, {2}}}},
		Tools:     api.Tools{{}, {}, {}},
		Stream:    &stream,
		Options:   map[string]any{"num_ctx": float64(4096), "num_gpu": float64(0)},
		KeepAlive: &api.Duration{Duration: 90 * time.Second},
	})
	s := meta.Shape
	if s.Surface != "openai" || s.Stream || s.Messages != 1 || s.Images != 2 || s.Tools != 3 ||
		s.NumCtx == nil || *s.NumCtx != 4096 || s.NumGPU == nil || *s.NumGPU != 0 || s.KeepAliveS == nil || *s.KeepAliveS != 90 {
		t.Fatalf("shape = %+v", *s)
	}
	if s.Client == "" {
		t.Error("no client id")
	}
}
