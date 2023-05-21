from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Breast_Cancer(BaseModel):
    clump_thickness: float
    uniform_cell_size: float
    uniform_cell_shape: float
    marginal_adhesion: float
    single_epithelial_size: float
    bare_nuclei: float
    bland_chromatin: float
    normal_nucleoli: float
    mitoses: float