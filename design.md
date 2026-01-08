对比结果固化成**结构化中间产物**（shared state），下游只吃“结论 + 关键参数 + 可追溯依据”。

**不要传递全部上下文，传递“可验证的决策记录”。**

---

## 推荐架构：Orchestrator + 2 个 Specialist（最稳）

### A. Orchestrator（编排/控上下文/控工具）

职责：

1. 解析用户需求 → 形成标准化需求状态
2. 调用“选型 agent”产出决策 JSON
3. 根据决策 JSON 再调用“API agent”去文档检索并生成 request body
4. 做最后一致性检查（字段是否齐全、与选型参数是否一致）

**Orchestrator 自己尽量不吃大段文档，只吃两个 agent 的结构化输出。**

---

### B. Solution/Design Agent（只看 private knowledge）

职责：**选 index 算法 + 给出可落地参数**
输入：用户需求、约束（延迟/成本/召回/更新频率/数据规模/多模态/过滤需求等）
检索源：private knowledge（算法优劣、经验规则、历史案例）
输出：一个“决策记录”（JSON）+ 简短解释

关键点：它不负责 API 字段名，不负责 body 拼装，避免被文档细节污染注意力。

---

### C. API Composer Agent（只看 private documentation）

职责：**把决策记录变成合法的 index request body**
输入：Design Agent 的 JSON 决策记录 + 用户环境信息（region、project、权限等）
检索源：private documentation（endpoint、schema、必填字段、枚举值、示例）
输出：API request（endpoint + body + 说明）
可选：做 schema 校验（如果你能拿到 OpenAPI/JSON Schema，强烈建议自动校验）

---

## “上下文传递”的核心：共享状态（Shared State）长这样

你需要一个外部 state（数据库/kv/内存都行），每一步只把**必要字段**塞回模型上下文。

示例（你可以按你们业务调整）：

```json
{
  "user_requirements": {
    "data_type": "text|image|hybrid",
    "scale": {"docs": 20000000, "avg_tokens": 800},
    "qps": 150,
    "latency_p95_ms": 200,
    "freshness": "near-real-time",
    "filters": ["tenant_id", "lang", "date_range"],
    "budget": "medium",
    "constraints": ["must_support_multi_tenant", "compliance:pii"]
  },
  "candidate_algorithms": [
    {"name": "HNSW", "pros": ["high_recall"], "cons": ["memory_heavy"]},
    {"name": "IVF_PQ", "pros": ["cheap_memory"], "cons": ["recall_tradeoff"]}
  ],
  "decision": {
    "selected": "HNSW",
    "rationale": [
      "P95 200ms + 高召回优先",
      "近实时更新更友好"
    ],
    "index_params": {
      "metric": "cosine",
      "dim": 768,
      "hnsw": {"M": 32, "ef_construction": 200, "ef_search": 64},
      "sharding": {"num_shards": 12}
    },
    "expected_tradeoffs": {
      "memory": "high",
      "build_time": "medium",
      "recall": "high"
    }
  },
  "api_plan": {
    "endpoint_hint": "CreateIndex",
    "doc_version": "vX.Y"
  }
}
```

**Design Agent 只负责把 `decision` 填好**；API Agent 只负责把 `decision.index_params` 映射到真实 schema 字段。