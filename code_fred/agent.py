from code_fred.models.base_model import BaseModel

class Agent():
    def __init__(self, model: BaseModel):
        self.model = model
    
    