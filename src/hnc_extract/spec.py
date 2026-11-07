"""Field specifications: one YAML file per field, one directory per document type.

A field is either
  * an ENTITY field, whose extractions become instances that other fields are
    scoped to (e.g. `specimen`: each extracted specimen is a target for
    `carcinoma_status`), or
  * an ATTRIBUTE field, extracted once per instance of its scope.

`scope` is `document` or the name of an entity field. `depends_on` gates a field
on other fields of the SAME scope instance (e.g. `histology` runs only for
specimens whose `carcinoma_status` is positive or suspicious). Parent results
are passed to the child prompt as context, so the model knows exactly which
specimen / test it is describing.

Every field shares the item schema (evidence, justification, assertion, value)
and may add `category` (closed value set), named `attributes`, and
`temporality`. The pydantic output model is built from the spec and used both
for JSON-schema constrained decoding and for validating the model's answer.
Worked examples in the spec are validated against that model at load time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    create_model,
    field_validator,
    model_validator,
)

RESERVED = {
    "evidence",
    "justification",
    "assertion",
    "value",
    "category",
    "temporality",
}
SCOPE_DOCUMENT = "document"


def _canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttributeSpec(Strict):
    type: Literal["string", "enum"] = "string"
    values: dict[str, str] | None = None  # enum value -> description
    nullable: bool = False
    description: str

    @model_validator(mode="after")
    def _check(self) -> AttributeSpec:
        if self.type == "enum" and not self.values:
            raise ValueError("enum attribute needs `values`")
        if self.type == "string" and self.values:
            raise ValueError("string attribute must not have `values`")
        return self


class Dependency(Strict):
    field: str
    # Gate passes if ANY extraction of `field` (same scope instance) has one of
    # these categories. None: passes if `field` extracted anything at all.
    categories: list[str] | None = None


class Example(Strict):
    excerpt: str
    target: str | None = None  # how the scope instance would be described
    output: dict
    note: str | None = None  # rationale for spec authors; not shown to the model


class FieldSpec(Strict):
    name: str
    version: int
    doc_type: str
    kind: Literal["entity", "attribute"]
    order: int = (
        1000  # tie-break for run/display order among fields whose dependencies are met
    )
    scope: str = SCOPE_DOCUMENT
    depends_on: list[Dependency] = []
    # Gate on attributes of the scope instance itself, e.g. {test_type: [pdl1_ihc]}.
    scope_filter: dict[str, list[str]] = {}
    title: str  # short human-readable name, used in prompts and the review sheet
    instruction: str
    value_description: str
    categories: dict[str, str] | None = None  # category -> description
    attributes: dict[str, AttributeSpec] = {}
    temporality: bool = False
    max_items: int | None = None
    # Only for entity fields: shown to child prompts to restrict them to one instance.
    target_instruction: str | None = None
    examples: list[Example] = []

    # Filled by the loader.
    fingerprint: str = ""
    _output_model: Any = PrivateAttr(default=None)

    @field_validator("attributes")
    @classmethod
    def _no_reserved(cls, v: dict) -> dict:
        clash = RESERVED & set(v)
        if clash:
            raise ValueError(
                f"attribute names clash with reserved keys: {sorted(clash)}"
            )
        return v

    @model_validator(mode="after")
    def _check(self) -> FieldSpec:
        if self.kind == "entity" and not self.target_instruction:
            raise ValueError(f"{self.name}: entity fields need `target_instruction`")
        if self.kind == "attribute" and self.target_instruction:
            raise ValueError(
                f"{self.name}: `target_instruction` is only for entity fields"
            )
        return self

    # ---- output model -------------------------------------------------------

    def output_model(self) -> type[BaseModel]:
        if self._output_model is None:
            self._output_model = _build_output_model(self)
        return self._output_model

    def json_schema(self) -> dict:
        return self.output_model().model_json_schema()

    def dependencies(self) -> list[str]:
        deps = [d.field for d in self.depends_on]
        if self.scope != SCOPE_DOCUMENT:
            deps.insert(0, self.scope)
        return deps


class Temporality(Strict):
    frame: Annotated[
        Literal["current", "historical", "unknown"],
        Field(description="Time frame of the value AS OF THIS DOCUMENT."),
    ]
    expression: Annotated[
        str | None,
        Field(description="Verbatim phrase that times the value; null if none."),
    ]


def _literal(values: list[str]):
    return Literal[tuple(values)]  # type: ignore[valid-type]


def _build_output_model(spec: FieldSpec) -> type[BaseModel]:
    item_fields: dict[str, Any] = {
        "evidence": (
            list[str],
            Field(
                min_length=1,
                description="Verbatim quote(s) copied character-for-character from the document.",
            ),
        ),
        "justification": (
            str,
            Field(description="Why the evidence supports this value."),
        ),
        "assertion": (
            Literal["affirmed", "uncertain"],
            Field(
                description=(
                    "affirmed: stated as fact (including 'consistent with'); "
                    "uncertain: hedged, possible, or a differential."
                )
            ),
        ),
        "value": (str, Field(description=spec.value_description)),
    }
    if spec.categories:
        item_fields["category"] = (
            _literal(list(spec.categories)),
            Field(description="Closed category for the value."),
        )
    for name, a in spec.attributes.items():
        t: Any = _literal(list(a.values)) if a.type == "enum" else str
        if a.nullable:
            t = t | None
        item_fields[name] = (t, Field(description=a.description))
    if spec.temporality:
        item_fields["temporality"] = (
            Temporality,
            Field(description="Temporal framing."),
        )

    cfg = ConfigDict(extra="forbid")
    camel = "".join(p.title() for p in spec.name.split("_"))
    item = create_model(f"{camel}Item", __config__=cfg, **item_fields)
    extractions_kw: dict[str, Any] = {
        "description": "One entry per distinct value; empty to abstain."
    }
    if spec.max_items is not None:
        extractions_kw["max_length"] = spec.max_items
    return create_model(
        f"{camel}Output",
        __config__=cfg,
        extractions=(list[item], Field(**extractions_kw)),
        summary=(
            str,
            Field(description="Brief reasoning: what was extracted or why abstained."),
        ),
    )


# ---- loading ----------------------------------------------------------------


class SpecSet:
    """All field specs for one document type, validated and topologically ordered."""

    def __init__(self, doc_type: str, specs: list[FieldSpec], instructions: dict):
        self.doc_type = doc_type
        self.instructions = instructions
        self.by_name = {s.name: s for s in specs}
        if len(self.by_name) != len(specs):
            raise ValueError(f"{doc_type}: duplicate field names")
        self._validate()
        self.order = self._topo_order()

    def __iter__(self):
        return (self.by_name[n] for n in self.order)

    def __getitem__(self, name: str) -> FieldSpec:
        return self.by_name[name]

    def _validate(self) -> None:
        for s in self.by_name.values():
            if s.doc_type != self.doc_type:
                raise ValueError(
                    f"{s.name}: doc_type {s.doc_type!r} != {self.doc_type!r}"
                )
            if s.scope != SCOPE_DOCUMENT:
                p = self.by_name.get(s.scope)
                if p is None or p.kind != "entity":
                    raise ValueError(
                        f"{s.name}: scope {s.scope!r} is not an entity field"
                    )
            if s.scope_filter:
                if s.scope == SCOPE_DOCUMENT:
                    raise ValueError(f"{s.name}: scope_filter needs an entity scope")
                ent = self.by_name[s.scope]
                for attr, vals in s.scope_filter.items():
                    a = ent.attributes.get(attr)
                    if a is None or a.type != "enum" or set(vals) - set(a.values or {}):
                        raise ValueError(
                            f"{s.name}: scope_filter {attr}={vals} invalid for {ent.name}"
                        )
            for d in s.depends_on:
                dep = self.by_name.get(d.field)
                if dep is None:
                    raise ValueError(f"{s.name}: depends on unknown field {d.field!r}")
                if dep.scope != s.scope:
                    raise ValueError(
                        f"{s.name}: dependency {d.field!r} must share scope {s.scope!r}"
                    )
                if d.categories:
                    bad = set(d.categories) - set(dep.categories or {})
                    if bad:
                        raise ValueError(
                            f"{s.name}: {d.field} has no categories {sorted(bad)}"
                        )
            model = s.output_model()
            for i, ex in enumerate(s.examples):
                try:
                    model.model_validate(ex.output)
                except Exception as e:  # noqa: BLE001
                    raise ValueError(
                        f"{s.name}: example {i} does not match the output schema: {e}"
                    ) from e

    def _topo_order(self) -> list[str]:
        done: list[str] = []
        remaining = dict(self.by_name)
        while remaining:
            ready = [
                s
                for s in remaining.values()
                if all(d in done for d in s.dependencies())
            ]
            if not ready:
                raise ValueError(
                    f"{self.doc_type}: dependency cycle among {sorted(remaining)}"
                )
            nxt = min(ready, key=lambda s: (s.order, s.name))
            done.append(nxt.name)
            del remaining[nxt.name]
        return done

    def children_of(self, entity: str) -> list[FieldSpec]:
        return [s for s in self if s.scope == entity]


def load_instructions(spec_dir: Path) -> dict:
    data = yaml.safe_load((spec_dir / "instructions.yaml").read_text())
    for k in ("version", "system", "user"):
        if k not in data:
            raise ValueError(f"instructions.yaml missing {k!r}")
    return data


def load_specs(spec_dir: Path, doc_type: str) -> SpecSet:
    instructions = load_instructions(spec_dir)
    inst_hash = _canonical_hash(instructions)
    specs = []
    for path in sorted((spec_dir / doc_type).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        try:
            spec = FieldSpec.model_validate(raw)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{path}: {e}") from e
        if spec.name != path.stem:
            raise ValueError(f"{path}: file name must match field name {spec.name!r}")
        # Any edit to the spec or the shared instructions invalidates cached calls.
        prompt_relevant = {k: v for k, v in raw.items() if k != "order"}
        spec.fingerprint = _canonical_hash(
            {"spec": prompt_relevant, "instructions": inst_hash}
        )
        specs.append(spec)
    if not specs:
        raise ValueError(f"no field specs in {spec_dir / doc_type}")
    return SpecSet(doc_type, specs, instructions)
