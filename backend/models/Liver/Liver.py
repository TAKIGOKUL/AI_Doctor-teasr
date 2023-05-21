from pydantic import BaseModel
class Liver(BaseModel):
    age: float
    gender: float
    total_bilirubin: float
    direct_bilirubin: float
    alkaline_phosphotase: float
    alamine_aminotransferase: float
    aspartate_aminotransferase: float
    total_protiens: float
    albumin: float
    albumin_and_globulin_ratio: float