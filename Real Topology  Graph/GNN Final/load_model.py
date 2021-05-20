"""
Code for loading MLP parameters

@author: hussein
"""

import torch
import MLP_Model

# load model using dict
FILE = "model.pth"
loaded_model = MLP_Model.MLP()
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)


row = [ -0.373219668865204, 2.17479133605957, 4.73384788632393,  -8.16998362541199]

print(row)

u1_predicted = MLP_Model.predict(row, loaded_model)

print(u1_predicted)