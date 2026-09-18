import torch
import torch.nn.functional as F    
import torch.nn as nn

#################### Wave Block Section##################
class WaveBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, n):
        super(WaveBlock, self).__init__()
        self.n = n
        self.init_conv = nn.Conv1d(in_channels, filters, kernel_size=1, padding=0)
        self.tanh_convs = nn.ModuleList()
        self.sigmoid_convs = nn.ModuleList()
        self.post_convs = nn.ModuleList()
        for i in range(n):
            dilation = 2 ** i
            pad = (kernel_size - 1) * dilation // 2
            self.tanh_convs.append(nn.Conv1d(filters, filters, kernel_size=kernel_size, padding=pad, dilation=dilation))
            self.sigmoid_convs.append(nn.Conv1d(filters, filters, kernel_size=kernel_size, padding=pad, dilation=dilation))
            self.post_convs.append(nn.Conv1d(filters, filters, kernel_size=1))

    def forward(self, x):
        x = self.init_conv(x)
        res_x = x
        for i in range(self.n):
            tanh_out = torch.tanh(self.tanh_convs[i](x))
            sigm_out = torch.sigmoid(self.sigmoid_convs[i](x))
            x = tanh_out * sigm_out
            x = self.post_convs[i](x)
            res_x = res_x + x
        return res_x

class EEGFeatureExtractor_waveblock(nn.Module):
    def __init__(self):
        super(EEGFeatureExtractor_waveblock, self).__init__()
        self.block1 = WaveBlock(1, 8, 3, 12)
        self.block2 = WaveBlock(8, 16, 3, 8)
        self.block3 = WaveBlock(16, 32, 3, 4)
        self.block4 = WaveBlock(32, 64, 3, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: [B, 1, T]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)  # shape: [B, C, 1]
        return x.squeeze(-1) # shape: [B, C]


# -------- 1D CNN BACKBONE (EEG time-domain) --------
class TemporalCNN1D(nn.Module):
    """
    Input:  x (B, C, T)
    Output: z (B, feat_dim)
    """
    def __init__(self, in_channels: int = 20, feat_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        # simple, strong baseline: Conv -> BN -> ReLU -> pool x3, then GAP and FC
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Linear(256, feat_dim)

    def forward(self, x):
        # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))  # (B,64,T)
        x = self.drop(x)
        x = F.max_pool1d(x, kernel_size=2)   # (B,64,T/2)

        x = F.relu(self.bn2(self.conv2(x)))  # (B,128,T/2)
        x = self.drop(x)
        x = F.max_pool1d(x, kernel_size=2)   # (B,128,T/4)

        x = F.relu(self.bn3(self.conv3(x)))  # (B,256,T/4)
        x = self.drop(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B,256)
        z = self.proj(x)  # (B, feat_dim)
        return z



# -------- EEGNet BACKBONE (depthwise + separable conv) --------
class EEGNet(nn.Module):
    def __init__(self, in_channels=8, T=10000, F1=8, D=2, F2=16,
                 dropout=0.5, feat_dim=128):
        super().__init__()
        # Block 1: Temporal conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 128), padding=(0, 64), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise conv over channels
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Block 2: Separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # Projection to feat_dim
        t_out = T // 4 // 8
        self.proj = nn.Linear(F2 * t_out, feat_dim)

    def forward(self, x):
        # x: (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.proj(x)


# -------- Lightweight 1D ResNet-ish BACKBONE --------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, dilation=1, padding=None):
        super().__init__()
        pad = ((kernel_size - 1)//2) * dilation if padding is None else padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = F.relu(out + identity)
        return out

class ResNet1D(nn.Module):
    """
    Input:  (B,C,T)
    Output: (B, feat_dim)
    """
    def __init__(self, in_channels=20, feat_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock1D(64, 128, stride=2)
        self.layer2 = ResidualBlock1D(128, 256, stride=2)
        self.layer3 = ResidualBlock1D(256, 256, stride=1, dilation=2)  # a touch of dilation
        self.proj   = nn.Linear(256, feat_dim)

    def forward(self, x):
        x = self.stem(x)                # (B,64,T)
        x = self.layer1(x)              # (B,128,T/2)
        x = self.layer2(x)              # (B,256,T/4)
        x = self.layer3(x)              # (B,256,~T/4)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B,256)
        z = self.proj(x)                # (B,feat_dim)
        return z


class ResNet1DGRU(nn.Module):
    """
    Multi-scale parallel conv + ResNet blocks + parallel GRU branch.
    Based on the 0.48 LB 1D EEG solution.
    Input:  (B, in_channels, T)
    Output: (B, feat_dim)
    """
    def __init__(self, in_channels=8, kernels=[3,5,7,9],
                 planes=24, fixed_kernel_size=5,
                 gru_hidden=128, feat_dim=256, num_resblocks=9,
                 input_T=2000):
        super().__init__()
        self.in_channels = in_channels
        self.planes = planes

        # Parallel multi-scale convs (one per kernel size)
        self.parallel_conv = nn.ModuleList([
            nn.Conv1d(in_channels, planes, kernel_size=k, stride=1, padding=0, bias=False)
            for k in kernels
        ])

        self.bn1  = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=False)

        # Strided conv to compress after concat
        self.conv1 = nn.Conv1d(planes, planes, fixed_kernel_size, stride=2, padding=2, bias=False)

        # ResNet blocks
        self.resblocks = nn.Sequential(*[
            ResidualBlock1D(planes, planes, stride=1, kernel_size=fixed_kernel_size,
                            padding=fixed_kernel_size // 2)
            for _ in range(num_resblocks)
        ])

        self.bn2     = nn.BatchNorm1d(planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

        # GRU branch on raw signal
        self.gru = nn.GRU(input_size=in_channels, hidden_size=gru_hidden,
                          num_layers=1, bidirectional=True, batch_first=True)

        # Projection: CNN features + GRU last hidden (bidirectional → 2*gru_hidden)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_T)
            outs = [conv(dummy) for conv in self.parallel_conv]
            out  = torch.cat(outs, dim=2)
            out  = self.bn1(out); out = self.relu(out)
            out  = self.conv1(out)
            out  = self.resblocks(out)
            out  = self.bn2(out); out = self.relu(out)
            out  = self.avgpool(out)
            cnn_out_dim = out.flatten(1).shape[1]
        self.proj = nn.Linear(cnn_out_dim + 2 * gru_hidden, feat_dim)

    def forward(self, x):
        # x: (B, C, T)
        # --- CNN branch ---
        outs = [conv(x) for conv in self.parallel_conv]
        out  = torch.cat(outs, dim=2)       # concat along time
        out  = self.bn1(out)
        out  = self.relu(out)
        out  = self.conv1(out)
        out  = self.resblocks(out)
        out  = self.bn2(out)
        out  = self.relu(out)
        out  = self.avgpool(out)
        cnn_feat = out.flatten(1)            # (B, planes * T')

        # --- GRU branch ---
        gru_in   = x.permute(0, 2, 1)       # (B, T, C)
        _, h_n   = self.gru(gru_in)         # h_n: (2, B, gru_hidden)
        gru_feat = h_n.permute(1, 0, 2).flatten(1)  # (B, 2*gru_hidden)

        # --- Concat + project ---
        combined = torch.cat([cnn_feat, gru_feat], dim=1)
        return self.proj(combined)


# -------- 2D CNN BACKBONE (Spectrogram) --------
class SmallCNN2D(nn.Module):
    """
    Input:  x (B, C, H, W)
    Output: z (B, feat_dim)
    """
    def __init__(self, in_channels: int = 3, feat_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8, W/8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B,256,1,1)
        )
        self.proj = nn.Linear(256, feat_dim)

    def forward(self, x):
        # x expected (B,C,H,W). If you have (B,C,W,H), permute before calling.
        x = self.features(x).flatten(1)  # (B,256)
        z = self.proj(x)                 # (B,feat_dim)
        return z