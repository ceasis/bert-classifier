%%writefile model.py

import torch
import torch.nn as nn

# Define your model architecture
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Function to load the model
def model_fn(model_dir):
    model = SimpleModel()
    model.load_state_dict(torch.load(f"{model_dir}/model.pth"))
    model.eval()
    return model

# Optional: Customize input/output handling (if needed)
def input_fn(request_body, request_content_type):
    import json
    data = json.loads(request_body)
    return torch.tensor(data)

def predict_fn(input_data, model):
    with torch.no_grad():
        return model(input_data).numpy()

def output_fn(prediction, response_content_type):
    import json
    return json.dumps(prediction.tolist())
