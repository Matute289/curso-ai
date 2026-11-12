from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class ExtraccionTecnica(BaseModel):
    """Información técnica extraída de un texto (log, arquitectura, etc.)."""

    tecnologias: List[str] = Field(
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos o "
            "herramientas mencionadas en el texto."
        ),
    )

    nivel_de_criticidad: Literal["baja", "media", "alta", "critica"] = Field(
        description=(
            "Qué tan grave o urgente es la situación descripta: "
            "baja, media, alta o critica."
        ),
    )

    resumen_tecnico: str = Field(
        description="Resumen técnico breve (1 o 2 oraciones) del texto.",
    )

    # Se valida con funciones (y no con min_length) para no agregar
    # restricciones al JSON schema que se envía al proveedor.
    @field_validator("tecnologias")
    @classmethod
    def _tecnologias_no_vacias(cls, value):
        if not value:
            raise ValueError("Debe haber al menos una tecnología.")
        return value

    @field_validator("resumen_tecnico")
    @classmethod
    def _resumen_con_contenido(cls, value):
        if len(value.strip()) < 10:
            raise ValueError("El resumen técnico es demasiado corto.")
        return value
