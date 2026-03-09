from pydantic import BaseModel,field_validator,model_validator
from typing import Optional
class ExperimentLogEntry(BaseModel):
    event:str
    experiment_id:str
    algorithm:str
    id: str
    start_time: float
    end_time:float
    worker_id:Optional[str]=""
    num_chunks:Optional[int] = 0
    workers:Optional[int] = 0 
    time: Optional[float] = None
    k:Optional[int] = 0
    iterations:Optional[int] =0

    @model_validator(mode='after')
    def compute_time(self) -> 'ExperimentLogEntry':
        if self.start_time is not None and self.end_time is not None:
            self.time = self.end_time - self.start_time
        else:
            raise ValueError("start_time and end_time must be set to compute time")
        return self