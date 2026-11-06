"""Field-DAG traversal over cached results, and the stage-wise batch runner.

`traverse()` recomputes the full node graph of one document from the store,
without calling any model. Each node is one (field, target instance) with a
status:
  done       final record with output (call status ok or flagged)
  failed     final record without output (call status invalid)
  pending    not yet run (or last attempt was a server error)
  gated_out  a depends_on condition is not met (e.g. histology for a
             specimen whose carcinoma_status is negative)
  blocked    its scope entity or a dependency is not done

`run()` processes fields in topological order. For each field it traverses
all documents, collects pending nodes, runs them concurrently against one
model, and persists each result as it completes. Because every stage is
persisted, a run can be interrupted and resumed, or run field by field.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from hnc_extract.documents import Document
from hnc_extract.evidence import EvidenceIndex
from hnc_extract.extract import CallRecord, run_call
from hnc_extract.llm import LLMClient, ModelConfig
from hnc_extract.prompting import PROMPT_FORMAT_VERSION, describe_item
from hnc_extract.spec import SCOPE_DOCUMENT, FieldSpec, SpecSet
from hnc_extract.store import Store, call_key

# Keys of an item that are passed downstream as context. Justification and
# temporality are left out: they don't identify the target and would make
# child cache keys churn on irrelevant wording changes.
_CONTEXT_DROP = {"justification", "temporality"}


@dataclasses.dataclass(frozen=True)
class Instance:
    path: str  # "" for the document; e.g. "specimen[2]" or "specimen[2]/margin[0]"
    field: str | None  # entity field that produced it; None for the document
    index: int | None
    item: dict | None
    parent: Instance | None

    def chain(self) -> list[Instance]:
        out, cur = [], self
        while cur is not None and cur.field is not None:
            out.append(cur)
            cur = cur.parent
        return out[::-1]


ROOT = Instance(path="", field=None, index=None, item=None, parent=None)


@dataclasses.dataclass
class Node:
    doc: Document
    spec: FieldSpec
    instance: Instance
    status: str
    key: str | None = None
    context: dict | None = None
    record: CallRecord | None = None
    reason: str | None = None

    @property
    def items(self) -> list[dict]:
        if self.status == "done" and self.record and self.record.output:
            return self.record.output["extractions"]
        return []


def _ctx_item(item: dict) -> dict:
    return {k: v for k, v in item.items() if k not in _CONTEXT_DROP}


def _child_path(parent: Instance, field: str, i: int) -> str:
    seg = f"{field}[{i}]"
    return f"{parent.path}/{seg}" if parent.path else seg


def traverse(
    doc: Document,
    specs: SpecSet,
    store: Store,
    model_fp: str,
    until: str | None = None,
) -> dict[tuple[str, str], Node]:
    nodes: dict[tuple[str, str], Node] = {}
    # entity field -> list of (parent instance, its child instances or None if not enumerable)
    entities: dict[str, list[tuple[Instance, list[Instance] | None]]] = {}

    for spec in specs:
        if spec.scope == SCOPE_DOCUMENT:
            groups: list[tuple[Instance, list[Instance] | None]] = [(ROOT, [ROOT])]
        else:
            groups = entities.get(spec.scope, [])

        produced: list[tuple[Instance, list[Instance] | None]] = []
        for parent, insts in groups:
            if insts is None:  # scope entity not enumerated for this parent
                nodes[(spec.name, parent.path + "/?")] = Node(
                    doc, spec, parent, "blocked", reason=f"{spec.scope} not done"
                )
                produced.append((parent, None))
                continue
            for inst in insts:
                node = _make_node(doc, spec, inst, insts, specs, nodes, store, model_fp)
                nodes[(spec.name, inst.path)] = node
                if spec.kind == "entity":
                    if node.status == "done":
                        children = [
                            Instance(
                                _child_path(inst, spec.name, i), spec.name, i, it, inst
                            )
                            for i, it in enumerate(node.items)
                        ]
                        produced.append((inst, children))
                    elif node.status == "gated_out":
                        produced.append((inst, []))
                    else:
                        produced.append((inst, None))
        if spec.kind == "entity":
            entities[spec.name] = produced
        if spec.name == until:
            break
    return nodes


def _make_node(
    doc: Document,
    spec: FieldSpec,
    inst: Instance,
    siblings: list[Instance],
    specs: SpecSet,
    nodes: dict[tuple[str, str], Node],
    store: Store,
    model_fp: str,
) -> Node:
    for attr, allowed in spec.scope_filter.items():
        got = (inst.item or {}).get(attr)
        if got not in allowed:
            return Node(
                doc, spec, inst, "gated_out", reason=f"{spec.scope}.{attr}={got}"
            )
    deps_ctx = []
    for d in spec.depends_on:
        dep = nodes.get((d.field, inst.path))
        if dep is None or dep.status in ("pending", "blocked", "failed"):
            why = "missing" if dep is None else dep.status
            return Node(doc, spec, inst, "blocked", reason=f"{d.field} {why}")
        if dep.status == "gated_out":
            return Node(doc, spec, inst, "gated_out", reason=f"{d.field} gated out")
        items = dep.items
        if d.categories is None:
            ok = bool(items)
        else:
            ok = any(it.get("category") in d.categories for it in items)
        if not ok:
            got = sorted({str(it.get("category")) for it in items}) or ["(none)"]
            return Node(
                doc, spec, inst, "gated_out", reason=f"{d.field}={','.join(got)}"
            )
        deps_ctx.append({"field": d.field, "items": [_ctx_item(it) for it in items]})

    chain_ctx = []
    for level in inst.chain():
        ent = specs[level.field]
        # siblings = other instances under the same parent
        sibs = siblings if level is inst else _siblings_of(level, nodes, ent)
        chain_ctx.append(
            {
                "field": ent.name,
                "title": ent.title,
                "target_instruction": ent.target_instruction,
                "item": _ctx_item(level.item),
                "siblings": [
                    describe_item(s.item) for s in sibs if s.path != level.path
                ],
            }
        )
    context = {"chain": chain_ctx, "deps": deps_ctx}
    key = call_key(
        doc_sha256=doc.text_sha256,
        spec_fingerprint=spec.fingerprint,
        model_fingerprint=model_fp,
        context=context,
        prompt_format=PROMPT_FORMAT_VERSION,
    )
    rec = store.get_final(key)
    if rec is None:
        return Node(doc, spec, inst, "pending", key=key, context=context)
    status = "done" if rec.output is not None else "failed"
    return Node(doc, spec, inst, status, key=key, context=context, record=rec)


def _siblings_of(
    level: Instance, nodes: dict[tuple[str, str], Node], ent: FieldSpec
) -> list[Instance]:
    parent = level.parent or ROOT
    node = nodes.get((ent.name, parent.path))
    if node is None:
        return []
    return [
        Instance(_child_path(parent, ent.name, i), ent.name, i, it, parent)
        for i, it in enumerate(node.items)
    ]


# ---- runner -------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(
    docs: Iterable[Document],
    specs: SpecSet,
    model: ModelConfig,
    client: LLMClient,
    store: Store,
    fields: set[str] | None = None,
    concurrency: int = 8,
    log: Callable[[str], None] = _log,
) -> dict[str, Counter]:
    docs = list(docs)
    model_fp = model.fingerprint()
    summary: dict[str, Counter] = {}
    for spec in specs:
        if fields and spec.name not in fields:
            continue
        pending: dict[str, Node] = {}
        n_nodes = Counter()
        for doc in docs:
            for node in traverse(doc, specs, store, model_fp, until=spec.name).values():
                if node.spec.name != spec.name:
                    continue
                n_nodes[node.status] += 1
                if node.status == "pending":
                    pending.setdefault(node.key, node)
        log(f"[{spec.name}] nodes {dict(n_nodes)}; {len(pending)} unique calls to run")
        results = Counter()
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futs = {
                pool.submit(_execute, n, specs, model, client): n
                for n in pending.values()
            }
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                store.put(rec)
                results[rec.status] += 1
                if i % 25 == 0 or i == len(futs):
                    log(
                        f"[{spec.name}] {i}/{len(futs)} {dict(results)} ({time.monotonic() - t0:.0f}s)"
                    )
        summary[spec.name] = results
    return summary


def _execute(
    node: Node, specs: SpecSet, model: ModelConfig, client: LLMClient
) -> CallRecord:
    return run_call(
        client=client,
        model=model,
        spec=node.spec,
        instructions=specs.instructions,
        doc=node.doc,
        context=node.context,
        key=node.key,
        evidence_index=EvidenceIndex(node.doc.text),
    )
