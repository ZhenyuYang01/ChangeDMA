from model.basic import *


class DMAEncoder(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=None, dims=None, drop_path_rate=0., head_init_scale=1.,
                 mem_dim=64, chunk_size=196, use_mem_batch_size=True, default_step_transform_max_lr=1e-1):
        super().__init__()
        dims = default(dims, [48, 96, 192, 384])
        depths = default(depths, [2, 2, 6, 2])

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], use_mem=(j==0), mem_dim=mem_dim, chunk_size=chunk_size,
                        use_mem_batch_size=use_mem_batch_size, accept_weight_residual=(i + j != 0),
                        drop_path=dp_rates[cur + j], default_step_transform_max_lr=default_step_transform_max_lr
                       ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        mem = None
        out_list = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            for blk in self.stages[i]:
                x, mem = blk(x, mem)
            out_list.append(x)
        return out_list

    def forward(self, x):
        x = self.forward_features(x)[-1]
        x = self.head(self.norm(x.mean([-2, -1])))
        return x


class Block(nn.Module):
    def __init__(self, dim, use_mem=True, mem_dim=64, chunk_size=64, use_mem_batch_size=True,
                 accept_weight_residual=False, drop_path=0.1, default_step_transform_max_lr=1e-1):
        super().__init__()
        # Local branch
        self.local_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.local_pw = nn.Sequential(
            LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            GlobalNorm(4 * dim),
            nn.Linear(4 * dim, dim)
        )
        # Global branch
        self.use_mem = use_mem
        if use_mem:
            self.proj_in = nn.Conv2d(dim, mem_dim, kernel_size=1)
            self.mem = NeuralMemory(dim=mem_dim, chunk_size=chunk_size,
                                    batch_size=(chunk_size*16) if use_mem_batch_size else None,
                                    qkv_receives_diff_views=False, accept_weight_residual=accept_weight_residual,
                                    attn_pool_chunks=True, qk_rmsnorm=True, momentum=True, momentum_order=1,
                                    default_step_transform_max_lr=default_step_transform_max_lr,
                                    per_parameter_lr_modulation=True)
            self.proj_out = nn.Conv2d(mem_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prev_weights=None):
        x_in = x
        x = self.local_dw(x)
        x = x.permute(0, 2, 3, 1) # B, H, W, C
        x = self.local_pw(x)
        x = x.permute(0, 3, 1, 2) # B, C, H, W

        if self.use_mem:
            b, c, h, w = x_in.size()
            x_mem = self.proj_in(x_in)
            x_mem = rearrange(x_mem, "b c h w -> b (h w) c").contiguous()
            x_mem, next_neural_mem_cache = self.mem(x_mem, prev_weights=prev_weights)
            mem_weight_residual = next_neural_mem_cache.updates
            x_mem = rearrange(x_mem, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            x_mem = self.proj_out(x_mem)
            x_mem = torch.sigmoid(x_mem)
            x = x_mem * x
        else:
            mem_weight_residual = prev_weights

        x = x_in + self.drop_path(x)
        return x, mem_weight_residual
