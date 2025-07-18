from collections import OrderedDict
import torch


odict_300w = torch.load("hrnetv2_pretrained/ver2/final_state_ver2.pth", weights_only=False)
correct_odict_300w = OrderedDict()
for key, value in odict_300w.items():
    dot_ndx = key.find('.')
    correct_key = key[dot_ndx+1:]
    correct_odict_300w[correct_key] = value
torch.save(correct_odict_300w, "model_ver2.pth")
