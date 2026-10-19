from model.basic import *
from model import Block


class Fusion(nn.Module):
    def __init__(self, img_size, embed_dims):
        super().__init__()
        self.mma = MMA([embed_dims[2], embed_dims[1], embed_dims[0]], embed_dims[0], img_size // 2)
        self.tem = nn.ModuleList([
            TSSCFM(embed_dims[0], depth=3),
            TSSCFM(embed_dims[1], depth=3),
            TSSCFM(embed_dims[2], depth=3),
            TSSCFM(embed_dims[3], depth=3),
        ])

    def forward(self, feature_list):
        x, enc3, enc2, enc1 = feature_list
        b = x.shape[0] // 2
        enc3, enc2, enc1 = self.mma([enc3, enc2, enc1])
        enc1 = self.tem[0](enc1[:b, :, :, :], enc1[b:, :, :, :])
        enc2 = self.tem[1](enc2[:b, :, :, :], enc2[b:, :, :, :])
        enc3 = self.tem[2](enc3[:b, :, :, :], enc3[b:, :, :, :])
        x = self.tem[3](x[:b, :, :, :], x[b:, :, :, :])
        return x, enc3, enc2, enc1


class MMA(nn.Module):  # Memory-guided Multi-scale Aggregation
    def __init__(self, in_dims, target_dim, target_size):
        super().__init__()
        self.target_dim = target_dim
        self.target_size = target_size
        self.projections = nn.ModuleList([nn.Conv2d(in_dim, target_dim, kernel_size=1) for in_dim in in_dims])
        self.ln1 = LayerNorm2d(target_dim * 3)
        self.drop_path = DropPath(0.05)
        self.channel = Block(target_dim * 3, chunk_size=1024)
        self.final_projections = nn.ModuleList([nn.Conv2d(target_dim, in_dim, kernel_size=1) for in_dim in in_dims])
        self.original_sizes = [target_size // 4, target_size // 2, target_size]

    def forward(self, features):
        upsampled_features = []
        output_features = []
        for i, feature in enumerate(features):
            feature = F.interpolate(feature, size=self.target_size, mode='bilinear', align_corners=False)
            feature = self.projections[i](feature)
            upsampled_features.append(feature)
        concatenated = torch.cat(upsampled_features, dim=1)
        attn_output = concatenated + self.drop_path(
            self.ln1(self.channel(concatenated)[0]))
        split_features = torch.split(attn_output, self.target_dim, dim=1)
        for i, split_feature in enumerate(split_features):
            split_feature = self.final_projections[i](split_feature)
            split_feature = F.interpolate(split_feature, size=self.original_sizes[i], mode='bilinear',
                                          align_corners=False)
            output_features.append(split_feature)
        return output_features


class TSSCFM(nn.Module):  # Temporal Interaction
    def __init__(self, dim, depth=1):
        super().__init__()
        self.layers = nn.Sequential(*[SCFMBlock(dim) for _ in range(depth)])

    def forward(self, x1, x2):
        for layer in self.layers:
            x1, x2 = layer(x1, x2)
        return x1 + x2


class SCFMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ca1 = ChannelAttention(dim, 4)
        self.ca2 = ChannelAttention(dim, 4)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        
    def forward(self, x1, x2):
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        c1 = self.ca1(x1)
        c2 = self.ca2(x2)
        x11 = x1 * c2
        x21 = x2 * c1
        s1 = self.sa1(x1)
        s2 = self.sa2(x2)
        x12 = x1 * s2
        x22 = x2 * s1        
        return x11+x12, x21+x22


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)








