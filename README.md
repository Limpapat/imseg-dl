# imseg-dl
multi-classes instance segmentation

# Installation & Example
- By pip
```bash
pip install "git+https://github.com/Limpapat/imseg-dl.git#egg=imsegdl"
```
Here is the pseudo-code to run the demo for imseg-dl package:
```python
from imsegdl.controller import ImsegDL

# Imseg setting up
ims = ImsegDL("/path/to/params.json",
              imformat="jpg",
              image_size=512,
              optimizer = 'adam',
              learning_rate = 0.001,
              init_weights = True,
              disp_plot=False, res_plot=True)

# Train & Evaluate the model
ims.train_eval_model
```

Hi 