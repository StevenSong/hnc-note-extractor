from typing import Annotated, Literal

from pydantic import BaseModel, Field

TEMPORALITY_FIELD = Field(
    description="The temporality information of the extracted value."
)
QUALIFIERS_FIELD = Field(
    description="Additional details to contextualize the extracted value.", min_length=1
)
EXTRACTIONS_FIELD = Field(
    description=(
        "Extracted value(s) with supporting information for the given field. "
        "Can have no extractions if the field is not present in the note."
    )
)
SUMMARY_FIELD = Field(description="Brief summary of reasoning for overall extraction.")


class Temporality(BaseModel):
    frame: Annotated[
        Literal["current", "historical", "unknown"],
        Field(
            description="Time frame of the extracted value as of this note, not when first established."
        ),
    ]
    expression: Annotated[
        str | None,
        Field(
            description="Verbatim span that times the value; null when the note gives no such phrase."
        ),
    ]


class _BaseExtractedItem(BaseModel):
    # consider ordering for autoregressive generation (potentially matters less for reasoning model)
    evidence: Annotated[
        list[str],
        Field(
            description=(
                "Verbatim span(s) from the note from which the value was extracted. "
                "Can have multiple pieces of evidence but must have at least one."
            ),
            min_length=1,
        ),
    ]
    justification: Annotated[
        str,
        Field(
            description="Justification for why the cited evidence supports the extraction."
        ),
    ]
    assertion: Annotated[
        Literal["affirmed", "uncertain"],
        Field(description="Whether the extracted value is affirmed, or uncertain."),
    ]
    value: Annotated[str, Field(description="extracted value")]


class _BaseFieldExtractions(BaseModel):
    pass


# fmt: off
class ExtractedItem(_BaseExtractedItem):
    pass
class ExtractedItemWithTemporality(_BaseExtractedItem):
    temporality: Annotated[Temporality, TEMPORALITY_FIELD]
class ExtractedItemWithQualifiers(_BaseExtractedItem):
    qualifiers: Annotated[list[str], QUALIFIERS_FIELD]
class ExtractedItemWithTemporalityAndQualifiers(_BaseExtractedItem):
    temporality: Annotated[Temporality, TEMPORALITY_FIELD]
    qualifiers: Annotated[list[str], QUALIFIERS_FIELD]

class FieldExtractions(_BaseFieldExtractions):
    extractions: Annotated[list[ExtractedItem], EXTRACTIONS_FIELD]
    summary: Annotated[str, SUMMARY_FIELD]
class FieldExtractionsWithTemporality(_BaseFieldExtractions):
    extractions: Annotated[list[ExtractedItemWithTemporality], EXTRACTIONS_FIELD]
    summary: Annotated[str, SUMMARY_FIELD]
class FieldExtractionsWithQualifiers(_BaseFieldExtractions):
    extractions: Annotated[list[ExtractedItemWithQualifiers], EXTRACTIONS_FIELD]
    summary: Annotated[str, SUMMARY_FIELD]
class FieldExtractionsWithTemporalityAndQualifiers(_BaseFieldExtractions):
    extractions: Annotated[list[ExtractedItemWithTemporalityAndQualifiers], EXTRACTIONS_FIELD]
    summary: Annotated[str, SUMMARY_FIELD]
# fmt: on


def get_schema(
    *,  # enforce kwargs
    require_temporality: bool,
    require_qualifiers: bool,
) -> type[_BaseFieldExtractions]:
    if require_temporality and require_qualifiers:
        return FieldExtractionsWithTemporalityAndQualifiers
    elif require_temporality and not require_qualifiers:
        return FieldExtractionsWithTemporality
    elif not require_temporality and require_qualifiers:
        return FieldExtractionsWithQualifiers
    elif not require_temporality and not require_qualifiers:
        return FieldExtractions
    else:
        raise ValueError(
            f"Unknown how to handle required configuration: "
            f"require_temporality={require_temporality}, "
            f"require_qualifiers={require_qualifiers}"
        )
