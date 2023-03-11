# imseg-dl
multi-classes instance segmentation

# Installation & Example
- By pip
```bash
pip install "git+https://github.com/Limpapat/imseg-dl.git#egg=imsegdl"
```
Then create a python file:
```python
from imsegdl.imsegdl import ImsegDL
imdl = ImsegDL("/path/to/params.json")
imdl.train_model()
# or
imdl.eval_model()
```
- By clone this repo
```bash
git clone https://github.com/Limpapat/imseg-dl.git
cd /path/to/imseg-dl
pip install -r ./requirements.txt
cd ./imsegdl
python3 train.py --params /path/to/params.json
# or
python3 eval.py --params /path/to/params.json
```

