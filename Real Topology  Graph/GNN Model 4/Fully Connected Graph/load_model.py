"""
Code for loading MLP parameters

@author: hussein
"""

import torch
import MLP_Model

# load model using dict
FILE = "model.pth"
loaded_model = MLP_Model.ModelE()
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

# for param in loaded_model.parameters():
#     print(param)

Input = [ 0,	0,	0,	0 ]

u = MLP_Model.predict(Input, loaded_model)

print(u)