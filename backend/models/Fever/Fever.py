from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Fever(BaseModel):
    cough: float
    # fever: float
    sore_throat: float
    shortness_of_breath: float
    head_ache: float
    age_60_and_above: float

   