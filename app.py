"""
K8s Cluster AI Triage Control Center
=====================================

A self-contained Streamlit application demonstrating how a Generative AI agent
can be safely embedded into an SRE incident-response workflow for a payment
processing namespace running on AWS EKS.

Purpose
-------
This is a portfolio / interview artifact for an SRE III role. It simulates:
    1. A live Kubernetes telemetry stream (synthetic, deterministic-ish).
    2. A "fault injection" control that produces a realistic, coupled
       failure signature: OOMKilled pod (exit code 137) + Istio sidecar
       503 upstream connect errors on dependent services.
    3. A context-aggregation layer that builds a structured payload out of
       raw telemetry (the same shape a real SRE would hand to an on-call
       engineer, or that a tool-using LLM agent would receive from a
       Prometheus/Loki/Kube API integration).
    4. An AI Agent layer (Anthropic Claude) that performs triage and emits
       a structured remediation plan -- with human-in-the-loop guardrails,
       since this operates in a regulated financial-services namespace.
    5. A two-pane operator console: live cluster state | AI diagnostics.

Design principles demonstrated (things an interviewer will look for):
    - Config is externalized (env vars / sidebar), never hardcoded secrets.
    - Every external call (AI API) is wrapped in explicit exception handling
      with typed fallbacks -- the dashboard must never crash because an LLM
      call failed or a key was missing.
    - The AI is treated as a *decision-support* tool, not an autonomous
      actor: it proposes commands, it does not execute them. This mirrors
      real-world change-management controls (four-eyes principle) expected
      in a bank's production environment.
    - Synthetic data generation is isolated from business logic so it can
      be swapped for a real Kubernetes/Prometheus/Istio client later.

Run
---
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."      # optional; app degrades gracefully without it
    streamlit run app.py
"""

from __future__ import annotations

import os
import json
import random
import logging
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from collections import deque
from typing import Any, Optional

import streamlit as st

# --------------------------------------------------------------------------
# Optional dependency: python-dotenv. Loads variables from a local .env file
# (see .env.example) into os.environ before AppConfig reads them. Imported
# defensively so the app still runs if the package isn't installed and the
# user is setting env vars some other way (shell export, CI secrets, etc.).
# --------------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# --------------------------------------------------------------------------
# Optional dependency: the Anthropic SDK. We import it defensively so that
# the rest of the dashboard (telemetry simulation, UI) still works even in
# an environment where the package isn't installed -- a common real-world
# situation during a live demo on an unfamiliar machine.
# --------------------------------------------------------------------------
try:
    import anthropic  # type: ignore
    _ANTHROPIC_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when SDK missing
    _ANTHROPIC_SDK_AVAILABLE = False


# ==========================================================================
# 1. LOGGING
# ==========================================================================
# Standard structured logging setup. In production this would ship to
# CloudWatch / Datadog; here it streams to stderr, which Streamlit surfaces
# in the terminal running `streamlit run`.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("k8s_ai_triage")


# ==========================================================================
# 2. CONFIGURATION MANAGEMENT
# ==========================================================================
# A single, immutable-by-convention config object. Values are sourced from
# environment variables first (12-factor style), with sidebar widgets
# allowed to override them for demo purposes only. Nothing here is ever
# written back to disk or logged verbatim (API key is masked).
@dataclass(frozen=True)
class AppConfig:
    cluster_name: str = os.environ.get("EKS_CLUSTER_NAME", "eks-prod-payments-usw2")
    aws_region: str = os.environ.get("AWS_REGION", "us-west-2")
    namespace: str = os.environ.get("K8S_NAMESPACE", "payment")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    max_tokens: int = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1500"))
    log_buffer_size: int = int(os.environ.get("LOG_BUFFER_SIZE", "40"))
    telemetry_tick_seconds: float = float(os.environ.get("TELEMETRY_TICK_SECONDS", "2.5"))

    def masked_key(self) -> str:
        """Never print a live API key to the UI or logs."""
        if not self.anthropic_api_key:
            return "(not configured)"
        return f"{self.anthropic_api_key[:7]}...{self.anthropic_api_key[-4:]}"


# ==========================================================================
# 3. DOMAIN MODEL: the payment namespace's synthetic service topology
# ==========================================================================
# A small, realistic dependency graph for a card-payments critical path.
# Keeping this declarative makes it trivial to swap for a real service-mesh
# discovery call (e.g. Istio's /config_dump or a service catalog API).
SERVICE_TOPOLOGY: dict[str, list[str]] = {
    "payment-api": ["payment-gateway", "fraud-detection"],
    "payment-gateway": ["ledger-service"],
    "fraud-detection": ["risk-scoring-engine"],
    "ledger-service": [],
    "risk-scoring-engine": [],
    "notification-service": ["payment-api"],
}

NODES = [
    "ip-10-0-3-142.ec2.internal",
    "ip-10-0-5-078.ec2.internal",
    "ip-10-0-7-233.ec2.internal",
]

POD_SUFFIX_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _fake_pod_suffix() -> str:
    return "".join(random.choice(POD_SUFFIX_ALPHABET) for _ in range(10))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ==========================================================================
# 4. MOCK TELEMETRY LOOP
# ==========================================================================
class TelemetryGenerator:
    """
    Produces synthetic, structured JSON log events for every service in the
    payment namespace. Under normal conditions it emits benign INFO/DEBUG
    request logs. It exposes `inject_failure()` to deterministically drop
    a coordinated OOMKill + Istio 503 incident into the stream, which is
    what the AI agent will later be asked to triage.

    Kept as a plain class (not a background thread) so it plays nicely with
    Streamlit's script-rerun execution model -- state lives in
    st.session_state, and `tick()` is called on every rerun.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def _normal_event(self) -> dict[str, Any]:
        service = random.choice(list(SERVICE_TOPOLOGY.keys()))
        node = random.choice(NODES)
        latency_ms = round(random.uniform(8.0, 120.0), 2)
        status_code = random.choices([200, 201, 400, 500], weights=[92, 4, 3, 1])[0]
        return {
            "timestamp": _now_iso(),
            "cluster": self.config.cluster_name,
            "namespace": self.config.namespace,
            "service": service,
            "pod": f"{service}-{random.randint(6, 9)}d{_fake_pod_suffix()[:4]}-{_fake_pod_suffix()[:5]}",
            "node": node,
            "level": "INFO" if status_code < 400 else "WARN",
            "event": "http_request",
            "http_status": status_code,
            "latency_ms": latency_ms,
            "message": f"{service} handled request in {latency_ms}ms",
        }

    def tick(self) -> dict[str, Any]:
        """Generate exactly one normal telemetry event."""
        try:
            return self._normal_event()
        except Exception:
            # Telemetry generation must never crash the dashboard; log and
            # emit a clearly-marked degraded event instead.
            logger.exception("Telemetry tick failed; emitting placeholder event")
            return {
                "timestamp": _now_iso(),
                "level": "ERROR",
                "event": "telemetry_generator_error",
                "message": "Synthetic telemetry generation failed internally.",
            }

    def inject_failure(self) -> list[dict[str, Any]]:
        """
        Produce the coordinated failure signature:
          1. payment-gateway pod is OOMKilled (exit code 137) on a node
             that is memory-pressured.
          2. Istio sidecars on the pod's callers (payment-api, ledger-service)
             immediately start reporting 503 UF/URX upstream connect errors,
             because the gateway pod is gone and the endpoint is unreachable.

        Returns the list of events (in causal order) so they can be pushed
        onto the shared log buffer and immediately visible to the operator.
        """
        try:
            node = random.choice(NODES)
            failed_pod = f"payment-gateway-7d4f9c6b78-{_fake_pod_suffix()[:5]}"
            events: list[dict[str, Any]] = []

            # --- Root cause: kubelet reports OOMKilled ---------------------
            events.append({
                "timestamp": _now_iso(),
                "cluster": self.config.cluster_name,
                "namespace": self.config.namespace,
                "service": "payment-gateway",
                "pod": failed_pod,
                "node": node,
                "container": "payment-gateway",
                "level": "CRITICAL",
                "event": "pod_terminated",
                "reason": "OOMKilled",
                "exit_code": 137,
                "memory_limit_mi": 512,
                "memory_working_set_mi": 511,
                "message": (
                    f"Container 'payment-gateway' in pod {failed_pod} was OOMKilled "
                    f"(exit code 137) on node {node}. Working set exceeded the 512Mi limit."
                ),
            })

            # --- Downstream symptom: Istio sidecar 503s on direct callers ---
            for caller in ["payment-api", "ledger-service"]:
                events.append({
                    "timestamp": _now_iso(),
                    "cluster": self.config.cluster_name,
                    "namespace": self.config.namespace,
                    "service": caller,
                    "node": random.choice(NODES),
                    "level": "ERROR",
                    "event": "istio_upstream_connect_error",
                    "http_status": 503,
                    "istio_flags": "UF,URX",
                    "upstream_cluster": "outbound|8443||payment-gateway.payment.svc.cluster.local",
                    "message": (
                        f"{caller}: upstream connect error or disconnect/reset before "
                        f"headers. reset reason: connection failure, transport failure "
                        f"reason: delayed connect error: 111 (Connection refused) "
                        f"-> 503 to payment-gateway.payment.svc.cluster.local"
                    ),
                })

            # --- Secondary symptom: fraud-detection sees elevated latency ---
            events.append({
                "timestamp": _now_iso(),
                "cluster": self.config.cluster_name,
                "namespace": self.config.namespace,
                "service": "fraud-detection",
                "node": random.choice(NODES),
                "level": "WARN",
                "event": "circuit_breaker_open",
                "message": (
                    "fraud-detection: circuit breaker OPEN for payment-api client "
                    "after 5xx error threshold exceeded (retry budget exhausted)."
                ),
            })

            return events
        except Exception:
            logger.exception("Failure injection failed")
            return [{
                "timestamp": _now_iso(),
                "level": "ERROR",
                "event": "failure_injection_error",
                "message": "Could not construct synthetic failure event set.",
            }]


# ==========================================================================
# 5. SRE CONTEXT BUILDER
# ==========================================================================
def build_context_payload(
    config: AppConfig,
    log_buffer: list[dict[str, Any]],
    failure_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate raw telemetry into the structured payload an SRE (or an LLM
    acting on an SRE's behalf) needs to triage an incident: the failing
    resource's identity, its node, its blast-radius neighbors in the
    dependency graph, and a bounded recent-log window for context.

    This function is intentionally pure (no I/O, no API calls) so it is
    trivially unit-testable in isolation from the AI integration.
    """
    if not failure_events:
        raise ValueError("build_context_payload requires at least one failure event")

    root_cause_event = next(
        (e for e in failure_events if e.get("event") == "pod_terminated"), failure_events[0]
    )
    affected_service = root_cause_event.get("service", "unknown-service")

    # Blast radius = services that directly depend on the failed service,
    # walked one hop out from the dependency graph declared above.
    direct_callers = [
        svc for svc, deps in SERVICE_TOPOLOGY.items() if affected_service in deps
    ]
    downstream_of_failed = SERVICE_TOPOLOGY.get(affected_service, [])

    # Keep only the most recent N log lines for prompt-size discipline --
    # a real implementation would also apply relevance filtering / sampling.
    recent_window = log_buffer[-config.log_buffer_size:]

    payload = {
        "meta": {
            "cluster": config.cluster_name,
            "aws_region": config.aws_region,
            "namespace": config.namespace,
            "generated_at": _now_iso(),
        },
        "root_cause_candidate": root_cause_event,
        "failure_event_sequence": failure_events,
        "topology": {
            "affected_service": affected_service,
            "direct_callers_upstream": direct_callers,
            "direct_dependencies_downstream": downstream_of_failed,
            "full_dependency_graph": SERVICE_TOPOLOGY,
        },
        "recent_logs_window": recent_window,
    }
    return payload


# ==========================================================================
# 6. AI AGENT INTEGRATION (Anthropic Claude)
# ==========================================================================
SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous Kubernetes Site Reliability co-pilot embedded in a
    tier-1 bank's production EKS platform, operating inside a regulated
    payments namespace. You are a DECISION-SUPPORT tool: you propose
    remediation, you never execute it. A human on-call engineer reviews and
    runs every command you suggest.

    You will be given a JSON telemetry payload describing a live incident.
    Respond ONLY in GitHub-flavored markdown with exactly these three
    second-level headings, in this order, and nothing before or after them:

    ## Incident Vector Identification
    Explain precisely what failed and the causal chain (root cause -> observed
    symptoms). Reference the specific pod, exit code, node, and error signatures
    from the payload.

    ## Cross-System Blast Radius Assessment
    Using the dependency topology in the payload, explain which upstream
    callers and downstream dependencies are impacted and how (e.g. request
    failures, circuit breakers, retry storms, potential data consistency
    risk for a ledger/payments system).

    ## Automated Remediation Workflow
    Give an ordered, numbered list of concrete steps. Each step that involves
    a command must include a fenced ```bash code block with the EXACT
    kubectl / istioctl / aws CLI command (correct flags, real-looking
    resource names taken from the payload). Since this is a regulated
    financial namespace, explicitly call out any step that should require a
    change-ticket / four-eyes approval before execution, and prefer safe,
    reversible actions (e.g. scaling, cordoning, rolling restart) over
    destructive ones. Do not suggest deleting PVCs, dropping databases, or
    disabling audit logging under any circumstances.

    Be concise, technically precise, and avoid generic filler advice.
""").strip()


class AIAgentError(Exception):
    """Raised for any recoverable failure in the AI triage call path."""


class K8sTriageAgent:
    """
    Thin wrapper around the Anthropic Messages API. Isolates all
    API-specific concerns (auth, retries-by-absence, error taxonomy) from
    the Streamlit UI layer, and provides a clearly-labeled offline fallback
    so the dashboard remains demoable without live credentials.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._client: Optional["anthropic.Anthropic"] = None

        if _ANTHROPIC_SDK_AVAILABLE and self.config.anthropic_api_key:
            try:
                self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            except Exception as exc:  # SDK init should never take down the app
                logger.exception("Failed to initialize Anthropic client")
                self._client = None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def triage(self, context_payload: dict[str, Any]) -> str:
        """
        Send the structured context payload to Claude and return markdown.
        Falls back to a deterministic simulated response (clearly labeled)
        if no live client is configured or the call fails for any reason --
        the operator console must never hard-crash mid-incident-review.
        """
        if not self.is_live:
            logger.warning("AI agent running in OFFLINE/SIMULATED mode (no API key or SDK).")
            return self._simulated_response(context_payload, reason="No Anthropic API key configured.")

        user_message = (
            "Live incident telemetry payload (JSON) for triage:\n\n"
            f"```json\n{json.dumps(context_payload, indent=2, default=str)}\n```"
        )

        try:
            response = self._client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=self.config.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
            markdown = "\n".join(text_blocks).strip()
            if not markdown:
                raise AIAgentError("Model returned an empty response body.")
            return markdown

        # --- Typed exception handling: distinguish failure classes so the
        # operator gets an actionable message instead of a stack trace. ---
        except anthropic.AuthenticationError as exc:
            logger.error("Anthropic authentication failed: %s", exc)
            return self._simulated_response(context_payload, reason="Authentication failed (check API key).")
        except anthropic.RateLimitError as exc:
            logger.error("Anthropic rate limit hit: %s", exc)
            return self._simulated_response(context_payload, reason="Rate limit exceeded; backing off.")
        except anthropic.APIConnectionError as exc:
            logger.error("Anthropic connection error: %s", exc)
            return self._simulated_response(context_payload, reason="Network error reaching Anthropic API.")
        except anthropic.APIStatusError as exc:
            logger.error("Anthropic API returned status error: %s", exc)
            return self._simulated_response(context_payload, reason=f"API error (status {exc.status_code}).")
        except AIAgentError as exc:
            logger.error("AI agent domain error: %s", exc)
            return self._simulated_response(context_payload, reason=str(exc))
        except Exception as exc:  # final safety net -- never propagate to UI
            logger.exception("Unexpected error during AI triage call")
            return self._simulated_response(context_payload, reason=f"Unexpected error: {exc}")

    @staticmethod
    def _simulated_response(context_payload: dict[str, Any], reason: str) -> str:
        """
        A deterministic, offline stand-in response used when live inference
        is unavailable. Clearly banner-labeled so it is never mistaken for a
        genuine model output during a demo or a real incident.
        """
        root = context_payload.get("root_cause_candidate", {})
        topo = context_payload.get("topology", {})
        pod = root.get("pod", "<unknown-pod>")
        node = root.get("node", "<unknown-node>")
        namespace = context_payload.get("meta", {}).get("namespace", "payment")
        callers = ", ".join(topo.get("direct_callers_upstream", [])) or "none detected"

        return textwrap.dedent(f"""
            > ⚠️ **SIMULATED AI RESPONSE** — live inference unavailable ({reason}).
            > Configure `ANTHROPIC_API_KEY` in the sidebar to enable real triage.

            ## Incident Vector Identification
            Pod `{pod}` on node `{node}` was terminated with exit code **137 (OOMKilled)**.
            The container exceeded its configured memory limit, causing the kubelet to
            invoke the OOM killer. Istio sidecars on dependent services immediately began
            reporting `503 UF,URX` upstream connect errors against the now-unreachable pod IP.

            ## Cross-System Blast Radius Assessment
            Direct upstream callers impacted: **{callers}**. These services will see
            elevated 5xx rates, exhausted retry budgets, and open circuit breakers until
            a healthy `payment-gateway` endpoint is available in the Kubernetes Service
            endpoint list. In namespace `{namespace}`, this directly affects the
            transaction-authorization critical path.

            ## Automated Remediation Workflow
            1. **Confirm current pod/replica state** (read-only, no approval needed):
               ```bash
               kubectl get pods -n {namespace} -l app=payment-gateway -o wide
               kubectl describe pod {pod} -n {namespace}
               ```
            2. **Scale out remaining healthy replicas to absorb load** (safe, reversible):
               ```bash
               kubectl scale deployment payment-gateway -n {namespace} --replicas=6
               ```
            3. **Raise the memory limit/request to prevent immediate recurrence**
               (⚠️ requires change-ticket approval — modifies resource guarantees):
               ```bash
               kubectl patch deployment payment-gateway -n {namespace} --type='json' \\
                 -p='[{{"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/memory","value":"1Gi"}}]'
               ```
            4. **Verify Istio endpoint health has recovered**:
               ```bash
               istioctl proxy-config endpoints deploy/payment-api.{namespace} \\
                 --cluster "outbound|8443||payment-gateway.{namespace}.svc.cluster.local"
               ```
            5. **If node-level memory pressure is suspected, cordon and drain the node**
               (⚠️ requires on-call lead approval — impacts co-located workloads):
               ```bash
               kubectl cordon {node}
               kubectl drain {node} --ignore-daemonsets --delete-emptydir-data --grace-period=60
               ```
            6. **Post-recovery validation**:
               ```bash
               kubectl rollout status deployment/payment-gateway -n {namespace}
               kubectl top pods -n {namespace} -l app=payment-gateway
               ```
        """).strip()


# ==========================================================================
# 7. STREAMLIT APPLICATION
# ==========================================================================
def _init_session_state(config: AppConfig) -> None:
    """Initialize all mutable state exactly once per browser session."""
    if "log_buffer" not in st.session_state:
        st.session_state.log_buffer = deque(maxlen=500)
    if "failure_active" not in st.session_state:
        st.session_state.failure_active = False
    if "failure_events" not in st.session_state:
        st.session_state.failure_events = []
    if "ai_response_md" not in st.session_state:
        st.session_state.ai_response_md = None
    if "context_payload" not in st.session_state:
        st.session_state.context_payload = None
    if "tick_count" not in st.session_state:
        st.session_state.tick_count = 0


def _render_sidebar(config: AppConfig) -> AppConfig:
    """Render configuration controls and return an (possibly overridden) config."""
    st.sidebar.title("⚙️ Cluster & Agent Configuration")

    st.sidebar.markdown("**Cluster context**")
    st.sidebar.code(
        f"cluster: {config.cluster_name}\n"
        f"region:  {config.aws_region}\n"
        f"namespace: {config.namespace}",
        language="yaml",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**AI Agent (Anthropic API)**")
    key_input = st.sidebar.text_input(
        "ANTHROPIC_API_KEY",
        value=config.anthropic_api_key,
        type="password",
        help="Only kept in this browser session's memory. Never logged or persisted.",
    )
    model_input = st.sidebar.text_input("Model", value=config.anthropic_model)

    st.sidebar.caption(f"Configured key: `{config.masked_key()}`")
    if not _ANTHROPIC_SDK_AVAILABLE:
        st.sidebar.warning("`anthropic` SDK not installed — running in simulated mode only.")

    # Return a new config with sidebar overrides applied (frozen dataclass,
    # so we reconstruct rather than mutate -- avoids accidental shared state).
    return AppConfig(
        cluster_name=config.cluster_name,
        aws_region=config.aws_region,
        namespace=config.namespace,
        anthropic_api_key=key_input,
        anthropic_model=model_input,
        max_tokens=config.max_tokens,
        log_buffer_size=config.log_buffer_size,
        telemetry_tick_seconds=config.telemetry_tick_seconds,
    )


def _severity_badge(level: str) -> str:
    return {
        "INFO": "🟢",
        "WARN": "🟡",
        "ERROR": "🟠",
        "CRITICAL": "🔴",
    }.get(level, "⚪")


def _render_telemetry_panel(generator: TelemetryGenerator, config: AppConfig) -> None:
    st.subheader("📡 Live Cluster Telemetry")

    col_a, col_b, col_c, col_d = st.columns(4)
    healthy = sum(1 for e in st.session_state.log_buffer if e.get("level") in ("INFO", "WARN"))
    critical = sum(1 for e in st.session_state.log_buffer if e.get("level") == "CRITICAL")
    errors = sum(1 for e in st.session_state.log_buffer if e.get("level") == "ERROR")
    col_a.metric("Namespace", config.namespace)
    col_b.metric("Healthy Events", healthy)
    col_c.metric("Error Events", errors)
    col_d.metric("Critical Events", critical, delta="OOMKilled" if critical else None,
                 delta_color="inverse")

    status = "🔴 OUTAGE ACTIVE" if st.session_state.failure_active else "🟢 NOMINAL"
    st.markdown(f"**Cluster status:** {status}")

    # Manual tick control -- keeps the demo deterministic and interviewer-
    # controllable rather than relying on background threads inside Streamlit,
    # which is an anti-pattern for this execution model.
    tick_col, inject_col, clear_col = st.columns([1, 1, 1])
    with tick_col:
        if st.button("▶️ Advance Telemetry Tick"):
            try:
                event = generator.tick()
                st.session_state.log_buffer.append(event)
                st.session_state.tick_count += 1
            except Exception:
                logger.exception("Unhandled error advancing telemetry tick")
                st.error("Telemetry tick failed unexpectedly. See application logs.")

    with inject_col:
        if st.button("🔥 Inject Coordinated Outage", type="primary"):
            try:
                failure_events = generator.inject_failure()
                st.session_state.failure_events = failure_events
                st.session_state.log_buffer.extend(failure_events)
                st.session_state.failure_active = True
                st.session_state.ai_response_md = None  # invalidate stale diagnosis
                st.session_state.context_payload = None
            except Exception:
                logger.exception("Unhandled error injecting failure state")
                st.error("Failure injection failed unexpectedly. See application logs.")

    with clear_col:
        if st.button("🧹 Reset Cluster State"):
            st.session_state.log_buffer.clear()
            st.session_state.failure_active = False
            st.session_state.failure_events = []
            st.session_state.ai_response_md = None
            st.session_state.context_payload = None
            st.session_state.tick_count = 0

    st.caption(f"Ticks processed: {st.session_state.tick_count} | "
               f"Buffer size: {len(st.session_state.log_buffer)}/{st.session_state.log_buffer.maxlen}")

    st.markdown("**Recent log stream** (most recent first)")
    recent = list(st.session_state.log_buffer)[-15:][::-1]
    if not recent:
        st.info("No telemetry yet. Click **Advance Telemetry Tick** to start the stream.")
    else:
        for e in recent:
            badge = _severity_badge(e.get("level", ""))
            svc = e.get("service", "-")
            st.text(f"{badge} [{e.get('level','?'):>8}] {e.get('timestamp','')} "
                     f"svc={svc:<22} {e.get('message','')[:110]}")

    with st.expander("🕸️ Service dependency graph (payment namespace)"):
        for svc, deps in SERVICE_TOPOLOGY.items():
            arrow = " → " + ", ".join(deps) if deps else " (leaf service)"
            st.text(f"{svc}{arrow}")


def _render_ai_panel(agent: K8sTriageAgent, config: AppConfig) -> None:
    st.subheader("🤖 AI Agent Diagnostics Console")

    if not st.session_state.failure_active:
        st.info("Cluster is nominal. Trigger **Inject Coordinated Outage** to generate "
                "an incident for the AI agent to triage.")
        return

    mode = "🟢 LIVE (Anthropic API)" if agent.is_live else "🟡 SIMULATED (offline fallback)"
    st.markdown(f"**Agent mode:** {mode}  |  **Model:** `{config.anthropic_model}`")

    if st.button("🩺 Run AI Triage on Current Incident", type="primary"):
        try:
            context_payload = build_context_payload(
                config=config,
                log_buffer=list(st.session_state.log_buffer),
                failure_events=st.session_state.failure_events,
            )
            st.session_state.context_payload = context_payload
        except ValueError as exc:
            st.error(f"Could not build triage context: {exc}")
            return
        except Exception:
            logger.exception("Unexpected error building context payload")
            st.error("Unexpected internal error while assembling SRE context. See logs.")
            return

        with st.spinner("Agent analyzing telemetry, dependency graph, and failure signature..."):
            try:
                markdown_response = agent.triage(context_payload)
                st.session_state.ai_response_md = markdown_response
            except Exception:
                # Belt-and-suspenders: K8sTriageAgent.triage() already catches
                # everything internally, but the UI layer never trusts that.
                logger.exception("Unhandled exception surfaced from AI agent triage()")
                st.session_state.ai_response_md = (
                    "> ⚠️ The AI agent raised an unhandled error. Falling back to "
                    "manual runbook procedures. Please check application logs."
                )

    if st.session_state.context_payload is not None:
        with st.expander("📦 Structured SRE context payload sent to the agent"):
            st.json(st.session_state.context_payload)

    if st.session_state.ai_response_md:
        st.markdown("---")
        st.markdown(st.session_state.ai_response_md)


def main() -> None:
    st.set_page_config(
        page_title="K8s Cluster AI Triage Control Center",
        page_icon="🛰️",
        layout="wide",
    )

    base_config = AppConfig()
    _init_session_state(base_config)

    st.title("🛰️ K8s Cluster AI Triage Control Center")
    st.caption(
        "AWS EKS · Istio Service Mesh · Payment Namespace — "
        "AI-assisted incident triage with human-in-the-loop remediation controls."
    )

    config = _render_sidebar(base_config)
    generator = TelemetryGenerator(config)
    agent = K8sTriageAgent(config)

    left, right = st.columns([1, 1], gap="large")
    with left:
        _render_telemetry_panel(generator, config)
    with right:
        _render_ai_panel(agent, config)

    st.markdown("---")
    st.caption(
        "Demo artifact only. The AI agent proposes remediation; it does not execute "
        "cluster-mutating commands. All destructive or resource-limit-changing steps "
        "are explicitly flagged as requiring change-management approval, consistent "
        "with production controls in a regulated financial-services environment."
    )


if __name__ == "__main__":
    main()
