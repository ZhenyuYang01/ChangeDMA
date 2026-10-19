from utils import *
from model.basic import *
from model.encoder import DMAEncoder
from model.fusion import Fusion


class ChangeDMA(nn.Module):
    def __init__(self, img_size=256, num_classes=1, dma_depths=None, dims=None, mem_dim=96, mem_group=196):
        super().__init__()
        dma_depths = default(dma_depths, [2, 2, 6, 2])
        dims = default(dims, [48, 96, 192, 384])
        self.encoder = DMAEncoder(depths=dma_depths, dims=dims, mem_dim=mem_dim, chunk_size=mem_group)
        self.fusion = Fusion(img_size, dims)
        self.decoder = nn.ModuleList([
            UpBlock(dims[3], dims[2]),
            UpBlock(dims[2]*2, dims[1]),
            UpBlock(dims[1]*2, dims[0]),
            UpBlock(dims[0]*2, 24)
        ])
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        enc1, enc2, enc3, x = self.encoder.forward_features(torch.cat([x1, x2], dim=0))
        x, enc3, enc2, enc1 = self.fusion([x, enc3, enc2, enc1])
        dec3 = self.decoder[0](x)
        dec2 = self.decoder[1](torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder[2](torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder[3](torch.cat([dec1, enc1], dim=1))
        out = self.final_conv(dec0)
        return torch.sigmoid(out)
