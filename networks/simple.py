import torch
import torch.nn as nn
import sys
from .utils import knn_idx, gather_idx, GPC2


class Net(nn.Module):
    def __init__(self, LV=True, emb=32, K=32):
        super().__init__()

        # inc = 24 if LV else 12
        inc = 30 if LV else 15
        self.fc = nn.Linear(inc, emb)
        self.K = K
        self.LV = LV

    def forward(self, x):
        in_f = GPC2(x, self.K, LV=self.LV)
        return self.fc(in_f.max(dim=1)[0])




if __name__ =="__main__":

    from torchsummary import summary
    model = Net().cuda()
    input_shape = (512, 3)

    print(summary(model, (input_shape), batch_size=1, device="cuda"))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
