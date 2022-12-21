from pydantic import BaseModel, validator, root_validator

COLUMNS = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal'
]


class InputData(BaseModel):
    age: int = 50
    sex: int = 1
    cp: int = 2
    trestbps: int = 130
    chol: int = 240
    fbs: int = 0
    restecg: int = 1
    thalach: int = 150
    exang: int = 0
    oldpeak: float = 0.8
    slope: int = 1
    ca: int = 0
    thal: int = 0

    @root_validator(pre=True)
    def input_data_validator(cls, values_dict):
        if list(values_dict.keys()) != COLUMNS:
            raise ValueError(f"Incorrect columns. Should be like {COLUMNS}.")
        return values_dict

    @validator('age')
    def age_values(cls, value):
        if value < 0 or value > 125:
            raise ValueError("Incorrect 'age' value (should be in [0, 125])")
        return value

    @validator('sex')
    def sex_values(cls, value):
        if not (value == 1 or value == 0):
            raise ValueError("Incorrect 'sex' value (should be in {0, 1})")
        return value

    @validator('cp')
    def cp_values(cls, value):
        if not (value == 0 or value == 1 or value == 2 or value == 3):
            raise ValueError("Incorrect 'cp' value (should be in {0, 1, 2, 3})")
        return value

    @validator('trestbps')
    def trestbps_values(cls, value):
        if value < 0 or value > 250:
            raise ValueError("Incorrect 'trestbps' value (should be in [0, 250])")
        return value

    @validator('chol')
    def chol_values(cls, value):
        if value < 0:
            raise ValueError("Incorrect 'chol' value (should be non-negative)")
        return value

    @validator('fbs')
    def fbs_values(cls, value):
        if not (value == 0 or value == 1):
            raise ValueError("Incorrect 'fbs' value (should be in {0, 1})")
        return value

    @validator('restecg')
    def restecg_values(cls, value):
        if not (value == 0 or value == 1 or value == 2):
            raise ValueError("Incorrect 'restecg' value (should be in {0, 1, 2})")
        return value

    @validator('thalach')
    def thalach_values(cls, value):
        if value < 0 or value > 220:
            raise ValueError("Incorrect 'thalach' value (should be in [0, 220])")
        return value

    @validator('exang')
    def exang_values(cls, value):
        if not (value == 1 or value == 0):
            raise ValueError("Incorrect 'exang' value (should be in {0, 1})")
        return value

    @validator('oldpeak')
    def oldpeak_values(cls, value):
        if value < 0 or value > 10:
            raise ValueError("Incorrect 'oldpeak' value (should be in [0, 10])")
        return value

    @validator('slope')
    def slope_values(cls, value):
        if not (value == 0 or value == 1 or value == 2):
            raise ValueError("Incorrect 'slope' value (should be in {0, 1, 2})")
        return value

    @validator('ca')
    def ca_values(cls, value):
        if not (value == 0 or value == 1 or value == 2 or value == 3):
            raise ValueError("Incorrect 'ca' value (should be in {0, 1, 2, 3})")
        return value

    @validator('thal')
    def thal_values(cls, value):
        if not (value == 0 or value == 1 or value == 2):
            raise ValueError("Incorrect 'thal' value (should be in {0, 1, 2})")
        return value
