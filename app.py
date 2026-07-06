"""
IncidentOps — Autonomous Multi-Agent IT Incident Response Teammate
Powered by NVIDIA NIM (build.nvidia.com) · Orchestrated in pure Python · UI in Streamlit

Agents:
  1. Coordinator   — plans the investigation, delegates, writes the final report
  2. Log Analyst   — inspects system logs for error patterns
  3. Knowledge     — retrieves matching runbooks / past incidents (lightweight RAG)
  4. Action Agent  — proposes remediation; executes ONLY after human approval
"""

import json
import os
import time
from datetime import datetime

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "meta/llama-3.1-70b-instruct"  # served as an NVIDIA NIM endpoint

st.set_page_config(page_title="IncidentOps · Multi-Agent Incident Response",
                   page_icon="🛠️", layout="wide")


def get_api_key() -> str:
    try:
        return st.secrets["NVIDIA_API_KEY"]
    except Exception:
        return os.environ.get("NVIDIA_API_KEY", "")


@st.cache_resource
def get_client():
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=get_api_key())


# ---------------------------------------------------------------------------
# Mock operational environment (data sources & external systems)
# ---------------------------------------------------------------------------
INCIDENTS = {
    "INC-1042 · Payment API latency spike": {
        "service": "payment-service",
        "alert": "p99 latency on payment-service jumped from 180ms to 2400ms in the last 15 minutes. Error rate 4.2%.",
        "logs": """2026-07-06 10:02:11 payment-service WARN  db-pool: connection wait time 850ms (threshold 200ms)
2026-07-06 10:02:44 payment-service ERROR db-pool: timeout acquiring connection after 5000ms
2026-07-06 10:03:02 payment-service WARN  db-pool: active connections 100/100 (pool exhausted)
2026-07-06 10:03:15 payment-service ERROR PaymentController: request failed - SQLTimeoutException
2026-07-06 10:03:40 payment-service WARN  db-pool: connection wait time 1900ms
2026-07-06 10:04:01 payment-service ERROR db-pool: timeout acquiring connection after 5000ms
2026-07-06 10:04:22 orders-service  INFO  normal operation, latency 95ms
2026-07-06 10:04:50 payment-service ERROR PaymentController: request failed - SQLTimeoutException
2026-07-06 10:05:10 payment-service WARN  slow query detected: SELECT * FROM transactions WHERE ... (4200ms)""",
    },
    "INC-1043 · Checkout service crash-looping": {
        "service": "checkout-service",
        "alert": "checkout-service pods restarted 14 times in 10 minutes after deploy v2.8.1. Availability degraded to 71%.",
        "logs": """2026-07-06 11:14:03 checkout-service INFO  starting v2.8.1 (deploy id d-88213)
2026-07-06 11:14:09 checkout-service ERROR config: missing required env var STRIPE_WEBHOOK_SECRET
2026-07-06 11:14:09 checkout-service FATAL startup aborted, exiting with code 1
2026-07-06 11:14:31 k8s             WARN  Back-off restarting failed container checkout-service
2026-07-06 11:15:02 checkout-service INFO  starting v2.8.1 (deploy id d-88213)
2026-07-06 11:15:07 checkout-service ERROR config: missing required env var STRIPE_WEBHOOK_SECRET
2026-07-06 11:15:07 checkout-service FATAL startup aborted, exiting with code 1
2026-07-06 11:16:44 k8s             WARN  CrashLoopBackOff: checkout-service restart count 14""",
    },
    "INC-1044 · Disk usage critical on analytics node": {
        "service": "analytics-node-3",
        "alert": "Disk usage on analytics-node-3 reached 94% and is growing ~2%/hour. ETL jobs at risk within 3 hours.",
        "logs": """2026-07-06 09:10:00 analytics-node-3 WARN  disk /data usage 88%
2026-07-06 09:40:00 analytics-node-3 WARN  disk /data usage 90%
2026-07-06 10:10:00 analytics-node-3 WARN  disk /data usage 92%
2026-07-06 10:20:13 etl-runner      INFO  job nightly-aggregate wrote 41GB temp files to /data/tmp
2026-07-06 10:22:51 etl-runner      WARN  temp cleanup step skipped: lock file present since 2026-07-01
2026-07-06 10:40:00 analytics-node-3 CRIT  disk /data usage 94%""",
    },
}

RUNBOOKS = {
    "RB-07 Database connection pool exhaustion": {
        "keywords": ["db-pool", "connection", "timeout", "sqltimeout", "latency", "pool"],
        "body": "Symptoms: connection wait times rising, pool at max, SQLTimeoutException. "
                "Past incident INC-0871 (May 2026): root cause was a slow unindexed query saturating the pool. "
                "Fix: identify slow query, kill it, temporarily raise pool size 100→150, then add index. "
                "Safe immediate action: restart payment-service with pool size 150 (5 min, low risk).",
    },
    "RB-12 CrashLoopBackOff after deployment": {
        "keywords": ["crashloop", "restart", "deploy", "env var", "fatal", "startup", "config"],
        "body": "Symptoms: pods crash immediately after a new deploy, often config/env related. "
                "Past incident INC-0912: missing secret after deploy. "
                "Fix: rollback to previous stable version immediately, then fix the config and redeploy. "
                "Safe immediate action: rollback to last stable release (2 min, low risk).",
    },
    "RB-15 Disk usage critical": {
        "keywords": ["disk", "usage", "94%", "temp", "cleanup", "etl", "storage"],
        "body": "Symptoms: disk usage above 90% and climbing. Common cause: stale temp files from ETL jobs. "
                "Past incident INC-0955: stale lock file prevented temp cleanup for days. "
                "Fix: remove stale lock, run cleanup job to purge /data/tmp older than 24h. "
                "Safe immediate action: run cleanup script (frees ~35-45GB, 3 min, low risk).",
    },
    "RB-03 High error rate on upstream dependency": {
        "keywords": ["upstream", "5xx", "dependency", "circuit breaker"],
        "body": "Symptoms: elevated 5xx from a dependency. Fix: enable circuit breaker, fail over to secondary region.",
    },
    "RB-21 Certificate expiry": {
        "keywords": ["certificate", "tls", "ssl", "expired", "handshake"],
        "body": "Symptoms: TLS handshake failures. Fix: renew certificate via cert-manager, restart ingress.",
    },
}


def retrieve_runbooks(text: str, top_k: int = 2):
    """Lightweight retrieval: keyword-overlap scoring over the runbook library."""
    text_l = text.lower()
    scored = []
    for name, rb in RUNBOOKS.items():
        score = sum(1 for kw in rb["keywords"] if kw in text_l)
        if score > 0:
            scored.append((score, name, rb["body"]))
    scored.sort(reverse=True)
    return [(name, body) for _, name, body in scored[:top_k]]


# ---------------------------------------------------------------------------
# Agent layer — every agent is an NVIDIA NIM call with a specialist persona
# ---------------------------------------------------------------------------
AGENT_META = {
    "Coordinator":     {"icon": "🧭", "color": "blue"},
    "Log Analyst":     {"icon": "🔎", "color": "orange"},
    "Knowledge Agent": {"icon": "📚", "color": "violet"},
    "Action Agent":    {"icon": "⚙️", "color": "red"},
    "System":          {"icon": "🖥️", "color": "gray"},
}


def log_activity(agent: str, content: str):
    st.session_state.feed.append({
        "agent": agent,
        "content": content,
        "time": datetime.now().strftime("%H:%M:%S"),
    })


def call_agent(agent: str, system_prompt: str, user_prompt: str, temperature=0.2) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        max_tokens=700,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = resp.choices[0].message.content.strip()
    log_activity(agent, out)
    return out


def run_investigation(incident_name: str):
    inc = INCIDENTS[incident_name]
    log_activity("System", f"Incident triggered: **{incident_name}**\n\nAlert: {inc['alert']}")

    # 1 — Coordinator plans
    with st.spinner("Coordinator is planning the investigation..."):
        plan = call_agent(
            "Coordinator",
            "You are the Coordinator agent leading an IT incident response team of specialist AI agents "
            "(Log Analyst, Knowledge Agent, Action Agent). Given an alert, produce a short numbered "
            "investigation plan (3-4 steps max) stating which agent handles each step. Be concise.",
            f"Incident: {incident_name}\nAlert: {inc['alert']}",
        )

    # 2 — Log Analyst investigates
    with st.spinner("Log Analyst is inspecting system logs..."):
        diagnosis = call_agent(
            "Log Analyst",
            "You are the Log Analyst agent. Analyze the raw logs, identify the error pattern and the most "
            "likely root cause. Answer in 3 short bullet points: Pattern, Root cause hypothesis, Evidence.",
            f"Alert: {inc['alert']}\n\nRaw logs:\n{inc['logs']}",
        )

    # 3 — Knowledge Agent retrieves runbooks
    matches = retrieve_runbooks(inc["alert"] + " " + inc["logs"] + " " + diagnosis)
    rb_text = "\n\n".join(f"### {name}\n{body}" for name, body in matches) or "No matching runbook found."
    with st.spinner("Knowledge Agent is searching runbooks and past incidents..."):
        knowledge = call_agent(
            "Knowledge Agent",
            "You are the Knowledge Agent. You retrieved the runbooks below from the team's knowledge base. "
            "Summarize in 2-3 sentences which past incident this matches and what the documented fix is. "
            "Mention the runbook ID.",
            f"Diagnosis from Log Analyst:\n{diagnosis}\n\nRetrieved runbooks:\n{rb_text}",
        )

    # 4 — Action Agent proposes remediation (requires human approval)
    with st.spinner("Action Agent is preparing a remediation proposal..."):
        proposal_raw = call_agent(
            "Action Agent",
            "You are the Action Agent. Based on the diagnosis and the runbook, propose ONE immediate "
            "remediation action. Respond ONLY with a JSON object, no markdown fences, with keys: "
            '"action" (imperative, one line), "reason" (one sentence), "risk" (low/medium/high), '
            '"estimated_time" (e.g. "3 min"). You may NOT execute anything without human approval.',
            f"Diagnosis:\n{diagnosis}\n\nKnowledge summary:\n{knowledge}",
            temperature=0.1,
        )
    try:
        clean = proposal_raw.replace("```json", "").replace("```", "").strip()
        proposal = json.loads(clean[clean.find("{"): clean.rfind("}") + 1])
    except Exception:
        proposal = {"action": proposal_raw[:200], "reason": "See activity feed.",
                    "risk": "medium", "estimated_time": "unknown"}

    st.session_state.proposal = proposal
    st.session_state.diagnosis = diagnosis
    st.session_state.knowledge = knowledge
    st.session_state.stage = "awaiting_approval"


def execute_action():
    p = st.session_state.proposal
    log_activity("System", "✅ Human operator **approved** the action.")
    with st.spinner("Action Agent is executing the approved action..."):
        time.sleep(1.5)  # simulated execution against the (mock) external system
        log_activity("Action Agent",
                     f"Executed: **{p['action']}**\n\nStatus: SUCCESS · duration {p.get('estimated_time', 'n/a')} · "
                     "service metrics returning to baseline.")
    with st.spinner("Coordinator is writing the post-incident report..."):
        call_agent(
            "Coordinator",
            "You are the Coordinator agent. Write a brief post-incident report (under 120 words) with sections: "
            "Summary, Root cause, Action taken, Follow-up recommendation.",
            f"Incident: {st.session_state.active_incident}\nDiagnosis: {st.session_state.diagnosis}\n"
            f"Knowledge: {st.session_state.knowledge}\nAction executed: {p['action']}",
        )
    st.session_state.stage = "resolved"


def reject_action():
    log_activity("System", "🚫 Human operator **rejected** the proposed action. "
                           "Incident escalated to the on-call engineer with the full agent investigation attached.")
    st.session_state.stage = "escalated"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("feed", [])
    st.session_state.setdefault("stage", "idle")
    st.session_state.setdefault("active_incident", None)
    st.session_state.setdefault("proposal", None)


init_state()

with st.sidebar:
    st.title("🛠️ IncidentOps")
    st.caption("Autonomous multi-agent incident response — humans stay in control.")
    st.markdown("**Powered by NVIDIA NIM** · " + MODEL.split("/")[-1])

    if not get_api_key():
        st.error("Add your NVIDIA_API_KEY (build.nvidia.com) to Streamlit secrets or environment.")

    st.divider()
    st.subheader("Agent team")
    for name, meta in AGENT_META.items():
        if name != "System":
            st.markdown(f"{meta['icon']} **{name}**")
    st.divider()
    if st.button("Reset session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

st.title("Incident Response Console")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("1 · Active alerts")
    incident = st.radio("Select an incident to hand to the agent team:",
                        list(INCIDENTS.keys()), index=0,
                        disabled=st.session_state.stage not in ("idle", "resolved", "escalated"))
    st.info(INCIDENTS[incident]["alert"])

    if st.session_state.stage in ("idle", "resolved", "escalated"):
        if st.button("🚨 Run agent investigation", type="primary", use_container_width=True,
                     disabled=not get_api_key()):
            st.session_state.feed = []
            st.session_state.active_incident = incident
            st.session_state.stage = "investigating"
            run_investigation(incident)
            st.rerun()

    # Human oversight panel
    if st.session_state.stage == "awaiting_approval" and st.session_state.proposal:
        p = st.session_state.proposal
        st.subheader("2 · Human approval required")
        risk = str(p.get("risk", "medium")).lower()
        risk_badge = {"low": "🟢 LOW", "medium": "🟡 MEDIUM", "high": "🔴 HIGH"}.get(risk, "🟡 MEDIUM")
        st.warning(f"**Proposed action:** {p['action']}\n\n"
                   f"**Why:** {p.get('reason','')}\n\n"
                   f"**Risk:** {risk_badge} · **Est. time:** {p.get('estimated_time','?')}")
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve & execute", type="primary", use_container_width=True):
            execute_action()
            st.rerun()
        if c2.button("🚫 Reject & escalate", use_container_width=True):
            reject_action()
            st.rerun()

    if st.session_state.stage == "resolved":
        st.success("Incident resolved. Post-incident report available in the activity feed.")
    if st.session_state.stage == "escalated":
        st.error("Incident escalated to a human engineer with full agent context.")

with col_right:
    st.subheader("Agent activity feed  ·  full transparency")
    st.caption("Every reasoning step by every agent is logged here for observability and audit.")
    if not st.session_state.feed:
        st.markdown("> No activity yet. Trigger an investigation to watch the agent team work.")
    for entry in st.session_state.feed:
        meta = AGENT_META.get(entry["agent"], AGENT_META["System"])
        with st.chat_message(entry["agent"], avatar=meta["icon"]):
            st.markdown(f"**:{meta['color']}[{entry['agent']}]** · `{entry['time']}`")
            st.markdown(entry["content"])
