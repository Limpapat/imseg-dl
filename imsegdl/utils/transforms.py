import torch.nn.functional as F
import torch

class EdgeEnhancement: # TODO
  def __init__(self, filter:list=[[0,-1,0], [-1,5,-1], [0,-1,0]], in_channels:int=1, out_channels:int=1) -> object:
    self.kernel = torch.tensor(filter, dtype=torch.float32).unsqueeze(0).expand(out_channels, in_channels, 3, 3)

  def __call__(self, samples:torch.Tensor)-> torch.Tensor:
    x = torch.unsqueeze(samples.detach(), 0)
    return F.conv2d(x, self.kernel, stride=1, padding=1).squeeze()

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}()"