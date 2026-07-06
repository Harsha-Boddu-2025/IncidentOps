# IncidentOps — Autonomous Multi-Agent Incident Response Teammate

A primary **Coordinator agent** orchestrates three specialist agents (**Log Analyst**, **Knowledge Agent**, **Action Agent**) to investigate IT incidents end-to-end. Every agent runs on **NVIDIA NIM** inference endpoints (Llama 3.1 70B via build.nvidia.com). Actions execute **only after explicit human approval**, and every reasoning step is streamed to a live activity feed for full observability.

## The plan

### Step 1 — Get a free NVIDIA API key 
1. Go to https://build.nvidia.com and sign in (free, gives you API credits).
2. Open any model (e.g. `meta/llama-3.1-70b-instruct`) → **Get API Key**.
3. Copy the key (starts with `nvapi-`).

### Step 2 — Run locally 
```bash
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-xxxxxxxx        # Windows: set NVIDIA_API_KEY=nvapi-...
streamlit run app.py
```
Open http://localhost:8501, pick an incident, click **Run agent investigation**.

### Step 3 — Deploy free on Streamlit Community Cloud 
1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io → **New app** → select your repo, main file `app.py`.
3. In **Advanced settings → Secrets**, add:
   ```toml
   NVIDIA_API_KEY = "nvapi-xxxxxxxx"
   ```
4. Deploy. You get a public URL for the demo (https://incidentops.streamlit.app/).

### Step 4 — Demo
1. 4-agent team, powered by NVIDIA NIM.
2. Trigger **INC-1042 (Payment API latency)** → narrate the feed: Coordinator plans → Log Analyst finds pool exhaustion → Knowledge Agent cites runbook RB-07 and past incident → Action Agent proposes a fix with risk level.
3. Point at the approval panel: *"the agent cannot act without me"* → click **Approve** → execution + post-incident report appear.
4. Re-run once with **Reject** to show the escalation path (human-in-control).

## Requirement mapping 

| Challenge requirement | Where it lives |
|---|---|
| Primary agent coordinating specialists | Coordinator plans & delegates to 3 agents |
| Autonomous planning, reasoning, execution | Coordinator plan → multi-step pipeline → execution |
| Integrate data sources, tools, external systems | Log store, runbook knowledge base (RAG), remediation executor |
| Workflow automation & multi-step decisions | detect → investigate → retrieve → propose → approve → execute → report |
| Transparency, observability, human oversight | Live agent activity feed + mandatory Approve/Reject gate |
| NVIDIA AI technologies | All 4 agents inference on NVIDIA NIM endpoints |

## Architecture 

```
                 ┌─────────────────────────────┐
   Alert ──────► │  Coordinator (NVIDIA NIM)   │ ─────► Post-incident report
                 └───────┬─────────┬───────────┘
                         │         │
        ┌────────────────┘         └───────────────┐
        ▼                    ▼                     ▼
  Log Analyst          Knowledge Agent        Action Agent
  (NIM + log store)    (NIM + runbook RAG)    (NIM + executor)
                                                   │
                                          Human Approve/Reject
                                                   │
                                             Execute action
```

## Scaling story 
- Swap the mock log store for real observability APIs (Datadog/Prometheus).
- Swap keyword retrieval for NVIDIA NeMo Retriever / NV-Embed embeddings.
- Swap Python orchestration for the NVIDIA NeMo Agent Toolkit or LangGraph.
- Self-host NIM containers on GPU infrastructure for data-sovereign deployments.
