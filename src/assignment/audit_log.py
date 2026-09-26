"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Remember the input and start time until the matching output arrives.

        A caller processing concurrent requests from one user should provide a
        request_id so each request has its own pending entry.
        """
        key = str(request_id or user_id)
        self._open[key] = {
            "user_id": user_id,
            "request_id": request_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_perf": perf_counter(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete an interaction record and append it to the audit log."""
        key = str(request_id or user_id)
        pending = self._open.pop(key, {})
        started_perf = pending.get("started_perf")
        latency_ms = (
            round(max(0.0, perf_counter() - started_perf) * 1000, 3)
            if started_perf is not None
            else None
        )
        self.logs.append(
            {
                "request_id": request_id or pending.get("request_id"),
                "user_id": user_id,
                "started_at": pending.get("started_at"),
                "completed_at": utc_now_iso(),
                "input": pending.get("input"),
                "output": text,
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": latency_ms,
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
