from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field

# Updated entity extraction class with allowed node types
class Entities(BaseModel):
    """Identifying information about entities, mapped to allowed graph nodes."""

    machines: List[str] = Field(
        default=[],
        description="Machines mentioned in the input",
    )
    components: List[str] = Field(
        default=[],
        description="Components or elements of the machine",
    )
    subsystems: List[str] = Field(
        default=[],
        description="Subsystems related to components",
    )
    characteristics: List[str] = Field(
        default=[],
        description="Characteristics or attributes of the machine or components",
    )
    tools: List[str] = Field(
        default=[],
        description="Tools related to machine operation or maintenance",
    )
    generic_nodes: List[str] = Field(
        default=[],
        description="Catch-all for any unclassified elements",
    )
