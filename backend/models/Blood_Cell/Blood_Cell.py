from pydantic import BaseModel
class Blood_Cell(BaseModel):
    pelvic_incidence: float
    pelvic_tilt: float
    lumbar_lordosis_angle: float
    sacral_slope: float
    pelvic_radius: float
    grade_of_spondyolistesis: float
    diagnose: float