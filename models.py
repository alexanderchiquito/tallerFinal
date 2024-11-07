from pydantic import BaseModel

class Casa(BaseModel):
    barrio: str
    ciudad: str
    departamento: str
    nro_habitaciones: int
    nro_banos: int
    metros_cuadrados: float
