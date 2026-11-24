"""Local web app for adjudicating a gold draft (`gold-review`).

The browser gets one JSON bundle (documents, draft clusters, every model's answer
and evidence per field, allowed values) and edits a review state that the server
saves to `<dir>/verdicts.json` after each change. `gold-import` reads that state.

State file (`verdicts.json`):
  version             1
  draft_created_utc   the draft it belongs to (a re-drafted dir refuses old state)
  fields              {row_id: {"value": v, "reviewed": bool}}; v is "yes"/"no" for
                      present, a list for list fields, else a string or null
  comments            {"doc_id|key": text}
  added               items no model found:
                      [{doc_id, kind, key, spans: [[start, end]], values: {field: v},
                        comment}]

Only the standard library serves HTTP. The server binds to localhost, answers
only requests whose Host header is local, and accepts saves only as JSON.
"""

from __future__ import annotations

import json
import os
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from hnc_extract import gold

BUNDLE = "review_bundle.json"
STATE = "verdicts.json"
STATE_VERSION = 1
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


# ---------------------------------------------------------------- bundle


def ui_value(field: str, v):
    """Draft (canonical) value -> the review app's form; None for ABSENT."""
    if v == gold.ABSENT:
        return None
    if v in (None, gold.NONE):
        return [] if field in gold.LIST_FIELDS else None
    if isinstance(v, (list, tuple)):
        return list(v)
    return v


def _item(views: dict, model: str, doc_id: str, kind: str, path: str) -> dict:
    key = "specimens" if kind == "specimen" else "tests"
    return next(x for x in views[model][doc_id][key] if x["path"] == path)


def build_bundle(
    meta: dict,
    records: list[dict],
    docs: list,
    views: dict[str, dict],
    options: dict[str, list[str]],
) -> dict:
    reasons = {
        r["row_id"]: r["reason"]
        for r in gold.review_rows(
            records, seed=meta["seed"], rate=meta["spot_check_rate"]
        )
    }
    out_records = []
    for r in records:
        evidence, desc = {}, {}
        for m, path in r["members"].items():
            it = _item(views, m, r["doc_id"], r["kind"], path)
            evidence[m] = it.get("field_evidence", {})
            if r["kind"] == "specimen":
                desc[m] = f"{it.get('label') or '?'}: {it.get('site') or ''}"
            else:
                desc[m] = f"{it.get('test')} [{it.get('specimen_label') or '-'}]"
        votes = []
        for v in r["votes"]:
            f = v["field"]
            votes.append(
                {
                    "field": f,
                    "answers": {m: ui_value(f, a) for m, a in v["answers"].items()},
                    "proposed": ui_value(f, v["proposed"]),
                    "agreed": v["agreed"],
                    "reason": reasons.get(f"{r['doc_id']}|{r['key']}|{f}"),
                }
            )
        out_records.append(
            {
                "doc_id": r["doc_id"],
                "kind": r["kind"],
                "key": r["key"],
                "members": r["members"],
                "spans": [list(s) for s in r["spans"]],
                "desc": desc,
                "evidence": evidence,
                "votes": votes,
            }
        )
    return {
        "meta": {k: v for k, v in meta.items() if k != "doc_ids"},
        "models": list(views),
        "fields": {"specimen": gold.SPECIMEN_FIELDS, "test": gold.TEST_FIELDS},
        "list_fields": sorted(gold.LIST_FIELDS),
        "text_fields": sorted(gold.TEXT_FIELDS),
        "options": options,
        "docs": [
            {"doc_id": d.doc_id, "patient_id": d.patient_id, "text": d.text}
            for d in docs
        ],
        "records": out_records,
    }


def write_bundle(directory: Path, bundle: dict) -> Path:
    path = directory / BUNDLE
    path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    return path


def read_bundle(directory: Path) -> dict:
    return json.loads((directory / BUNDLE).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- state


def empty_state(meta: dict) -> dict:
    return {
        "version": STATE_VERSION,
        "draft_created_utc": meta.get("created_utc"),
        "fields": {},
        "comments": {},
        "added": [],
    }


def read_state(directory: Path, meta: dict) -> dict:
    path = directory / STATE
    if not path.exists():
        return empty_state(meta)
    state = json.loads(path.read_text(encoding="utf-8"))
    check_state_shape(state, meta)
    return state


def check_state_shape(state: dict, meta: dict) -> None:
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        raise ValueError(f"not a version-{STATE_VERSION} review state")
    if state.get("draft_created_utc") != meta.get("created_utc"):
        raise ValueError(
            f"review state belongs to the draft of {state.get('draft_created_utc')}, "
            f"this dir holds the draft of {meta.get('created_utc')}"
        )
    for k, t in (("fields", dict), ("comments", dict), ("added", list)):
        if not isinstance(state.get(k), t):
            raise ValueError(f"review state: {k!r} must be a {t.__name__}")


def write_state(directory: Path, state: dict) -> None:
    """Atomic write; the previous file is kept as verdicts.json.bak."""
    path = directory / STATE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    if path.exists():
        shutil.copy2(path, path.with_suffix(".json.bak"))
    os.replace(tmp, path)


def state_verdicts(state: dict) -> tuple[dict[str, object], list[dict]]:
    """(verdicts {row_id: value}, added items) for gold.build_gold."""
    verdicts = {rid: e["value"] for rid, e in state["fields"].items() if "value" in e}
    return verdicts, list(state["added"])


def validate_state(
    state: dict, records: list[dict], options: dict[str, list[str]]
) -> list[str]:
    """Problems that would make the gold key wrong; empty if none."""
    errors = []
    known = {
        f"{r['doc_id']}|{r['key']}|{v['field']}": v["field"]
        for r in records
        for v in r["votes"]
    }
    for rid, e in state["fields"].items():
        if rid not in known:
            errors.append(f"{rid}: not in the draft")
        elif "value" in e:
            msg = gold.validate_value(known[rid], e["value"], options)
            if msg:
                errors.append(f"{rid}: {msg}")
    docs = {r["doc_id"] for r in records}
    for a in state["added"]:
        where = f"added {a.get('doc_id', '?')[:8]}|{a.get('key', '?')}"
        if a.get("doc_id") not in docs:
            errors.append(f"{where}: unknown doc_id")
            continue
        if a.get("kind") not in ("specimen", "test"):
            errors.append(f"{where}: kind must be specimen or test")
            continue
        vals = a.get("values", {})
        has_label = gold.norm_label(
            vals.get("label" if a["kind"] == "specimen" else "specimen_label")
        )
        if not a.get("spans") and not has_label:
            errors.append(
                f"{where}: needs a part label or selected evidence text, "
                "or scoring cannot match it"
            )
        for f, v in vals.items():
            msg = gold.validate_value(f, v, options)
            if msg:
                errors.append(f"{where}: {msg}")
    return errors


def progress(records: list[dict], reasons: dict[str, str], state: dict) -> dict:
    """Rows a person must decide, and how many are done.

    A row is required when the draft sent it to review; an if_present row only
    when its item's present answer is yes.
    """
    req = done = 0
    for r in records:
        pid = f"{r['doc_id']}|{r['key']}|present"
        pv = state["fields"].get(pid, {}).get("value", r["votes"][0]["proposed"])
        for v in r["votes"]:
            rid = f"{r['doc_id']}|{r['key']}|{v['field']}"
            if rid not in reasons or (v["field"] != "present" and pv != "yes"):
                continue
            req += 1
            done += bool(state["fields"].get(rid, {}).get("reviewed"))
    return {"required": req, "done": done}


# ---------------------------------------------------------------- server


def _index_html() -> bytes:
    return resources.files("hnc_extract").joinpath("review_ui/index.html").read_bytes()


class _Handler(BaseHTTPRequestHandler):
    server: _ReviewServer

    def log_message(self, fmt, *args):  # quiet
        pass

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return self.server.allow_any_host or host in LOCAL_HOSTS

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "host not allowed"})
        srv = self.server
        if self.path in ("/", "/index.html"):
            return self._send(HTTPStatus.OK, _index_html(), "text/html; charset=utf-8")
        if self.path == "/api/bundle":
            return self._send(HTTPStatus.OK, srv.bundle_bytes, "application/json")
        if self.path == "/api/state":
            with srv.lock:
                return self._json(HTTPStatus.OK, srv.state)
        if self.path == "/api/reports":
            with srv.lock:
                return self._json(HTTPStatus.OK, srv.summary())
        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "host not allowed"})
        if self.path != "/api/state":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "json only"})
        srv = self.server
        try:
            n = int(self.headers.get("Content-Length") or 0)
            state = json.loads(self.rfile.read(n))
            check_state_shape(state, srv.meta)
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        with srv.lock:
            write_state(srv.directory, state)
            srv.state = state
            out = srv.summary()
        self._json(HTTPStatus.OK, {"ok": True, **out})


class _ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, addr, directory: Path, bundle: dict, records, meta, allow_any_host
    ):
        import threading

        super().__init__(addr, _Handler)
        self.directory = directory
        self.bundle_bytes = json.dumps(bundle, ensure_ascii=False).encode()
        self.options = bundle["options"]
        self.records = records
        self.meta = meta
        self.state = read_state(directory, meta)
        self.lock = threading.Lock()
        self.allow_any_host = allow_any_host

    def summary(self) -> dict:
        """Validation errors and the report-level rollup implied by the answers."""
        errors = validate_state(self.state, self.records, self.options)
        try:
            verdicts, added = state_verdicts(self.state)
            g = gold.build_gold(self.records, verdicts, added)
            reports = {d: doc["report"] for d, doc in g.items()}
        except (
            Exception
        ) as e:  # noqa: BLE001 - a half-entered answer must not break saving
            errors.append(f"report rollup failed: {e}")
            reports = {}
        return {"errors": errors, "reports": reports}


def make_server(
    directory: Path,
    bundle: dict,
    records: list[dict],
    meta: dict,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> _ReviewServer:
    return _ReviewServer(
        (host, port),
        directory,
        bundle,
        records,
        meta,
        allow_any_host=host not in ("127.0.0.1", "localhost", "::1"),
    )
