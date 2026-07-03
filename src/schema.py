from typing import Annotated, Literal

from pydantic import BaseModel, Field


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


class ExtractedItem(BaseModel):
    # order matters for structured decoding (potentially less-so for reasoning model)
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
    temporality: Annotated[
        Temporality,
        Field(description="The temporality information of the extracted value."),
    ]
    value: Annotated[str, Field(description="extracted value")]


class FieldExtractions(BaseModel):
    extractions: Annotated[
        list[ExtractedItem],
        Field(
            description=(
                "Extracted value(s) with supporting information for the given field. "
                "Can have no extractions if the field is not present in the note."
            )
        ),
    ]
    summary: Annotated[
        str, Field(description="Brief summary of reasoning for overall extraction.")
    ]
