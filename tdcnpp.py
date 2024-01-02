
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, no_residual = False):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.PReLU(),
            nn.GroupNorm(out_channels, out_channels, eps=1e-8), 
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, groups=out_channels, padding=dilation * (kernel_size - 1) // 2),
            nn.PReLU(),
            nn.GroupNorm(out_channels, out_channels, eps=1e-8)
        )

        self.conv2_res = None if no_residual else nn.Conv1d(out_channels, in_channels, 1)
        self.conv2_skip = nn.Conv1d(out_channels, in_channels, 1)

    def forward(self, x0):
        x = self.conv_block(x0)

        if self.conv2_res is None:
            out = None
        else:
            out = self.conv2_res(x) + x0

        skip = self.conv2_skip(x)
        return skip, out

class MaskNetworkPlus(nn.Module):
    def __init__(self, enc_dim, enc_ker, conv_in_channels, conv_out_channels, conv_ker, n_blocks, n_repeats, num_sources, use_consistency):
        super(MaskNetworkPlus, self).__init__()
        self.enc_dim = enc_dim
        self.enc_ker = enc_ker
        self.conv_out_channels = conv_out_channels
        self.conv_ker = conv_ker
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.num_sources = num_sources
        self.conv_in_channels = conv_in_channels
        self.use_consistency = use_consistency

        self.in_norm = nn.GroupNorm(num_groups=1, num_channels=enc_dim, eps=1e-8)
        self.in_conv = nn.Conv1d(in_channels=enc_dim, out_channels=conv_in_channels, kernel_size=1)

        self.receptive_field = 0
        self.blocks = nn.ModuleList([])
        for i in range(n_repeats):
            for j in range(n_blocks):
                multi = 2**j
                self.blocks.append(ConvBlock(conv_in_channels, 
                                             conv_out_channels, 
                                             conv_ker, 
                                             dilation=multi, 
                                             no_residual=(j == (n_blocks - 1) and i == (n_repeats - 1))))

                self.receptive_field += conv_ker if i == 0 and j == 0 else (conv_ker-1) * multi
        
        self.prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(conv_in_channels, num_sources * enc_dim, 1, bias=False)

        # Dense connection in TDCNpp
        self.dense_skip = nn.ModuleList([nn.Conv1d(conv_in_channels, conv_in_channels, 1) for _ in range(n_repeats - 1)])

        scaling_param = torch.Tensor([0.9**l for l in range(0, n_blocks - 1)]) # -1 because we don't learn scaling for first block
        scaling_param = scaling_param.unsqueeze(0).expand(n_repeats, n_blocks - 1)
        self.scaling_param = nn.Parameter(scaling_param, requires_grad=True) 

        self.consistency = nn.Linear(conv_in_channels, num_sources) if use_consistency else None

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.in_norm(x)
        out = self.in_conv(x)

        residuals = 0.0
        i_repeat, i_block = 0, 0 # Easy way to keep track of repeats and blocks
        for i, block in enumerate(self.blocks):
            if i % self.n_blocks == 0:
                i_block = 0
                # Add dense connection to the input of the "next" repeat
                if i != 0:
                    out = out + self.dense_skip[i_repeat](input_of_repeat)
                    i_repeat += 1
                # Save input of the current repeat for dense connection
                input_of_repeat = out.clone() 
            # Residual scaling
            scale = self.scaling_param[i_repeat - 1, i_block - 1] if i_repeat > 0 else 1.0
            # As in ConvTasNet but with residuals scaling
            skip, out = block(out)
            residuals = residuals * scale
            residuals = residuals + skip
            i_block += 1

        out = self.prelu(residuals)
        masks = self.mask_conv(out)
        masks = torch.sigmoid(masks)

        mix_weights = self.consistency(out.mean(-1)) if self.use_consistency else None
        if mix_weights is not None:
            mix_weights = F.softmax(mix_weights, -1)

        return masks.view(batch_size, self.num_sources, self.enc_dim, -1), mix_weights


class TDCNpp(nn.Module):
    def __init__(self, enc_dim, enc_ker, conv_in_channels, conv_out_channels, conv_ker, n_blocks, n_repeats, num_sources, use_consistency = True):
        """
        The differences to ConvTasnet are:

        1. Channel wise layer norm instead of global -> instance norm?
        2. Longer-range skip-residual connections from earlier repeat inputs
           to later repeat inputs after passing them through dense layer.
        3. Learnable scaling parameter after each dense layer. The scaling
           parameter for the second dense  layer  in  each  convolutional
           block (which  is  applied  rightbefore the residual connection) is
           initialized to an exponentially decaying scalar equal to 0.9**L,
           where L is the layer or block index.

        enc_dim: Number of filters in autoencoder.
        enc_ker: Length of the filters in the autoencoder.
        conv_in_channels: Number of input channels in convolutional blocks. 
        conv_out_channels: Number of output channels in convolutional blocks.
        conv_ker: Kernel size in convolutional blocks.
        n_blocks: Number of convolutional blocks in each repeat.
        n_repeats: Number of repeats.
        num_sources: Number of sources to separate.
        """
        super(TDCNpp, self).__init__()
        self.enc_dim = enc_dim
        self.enc_ker = enc_ker
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_ker = conv_ker
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.num_sources = num_sources
        self.enc_stride = enc_ker//2
        self.use_consistency = use_consistency

        self.encoder = nn.Conv1d(1, enc_dim, enc_ker, stride=self.enc_stride, bias=False)

        self.mask_net = MaskNetworkPlus(enc_dim, enc_ker, conv_in_channels, conv_out_channels, conv_ker, n_blocks, n_repeats, num_sources, use_consistency)

        self.decoder = nn.ConvTranspose1d(enc_dim, 1, enc_ker, stride=self.enc_stride, bias=False)

    @staticmethod
    def _mixture_consistency_projection(separated_sources, original_mixture, mix_weights):
        """
        Projects the separated sources such that they sum up to the original mixture.

        Args:
        - separated_sources (torch.Tensor): The separated sources tensor of shape (batch_size, num_sources, T).
        - original_mixture (torch.Tensor): The original mixture tensor of shape (batch_size, 1, T).

        Returns:
        - torch.Tensor: The projected sources of shape (batch_size, num_sources, T).
        """

        # Calculate the residual between the original mixture of mixtures and the sum of the separated sources which is the estimated mixture of mixtures
        residual = original_mixture.squeeze(1) - separated_sources.sum(dim=1)

        if mix_weights is None:
            # Calculate mean residual
            correction = residual / separated_sources.size(1)
        else:
            # Calculate weighted residual
            correction = mix_weights.unsqueeze(-1) * residual.unsqueeze(1)
        
        # Add correction to each separated source
        projected_sources = separated_sources + correction

        return projected_sources

    # Taken from ConvTasNet implementation in torchaudio
    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            Tensor: Padded Tensor
            int: Number of paddings performed
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_ker % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings
    
    def forward(self, x):
        padded, num_pads = self._align_num_frames_with_strides(x)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]

        transformed = self.encoder(padded)
        masks, mix_weights = self.mask_net(transformed)
        masked = masks * transformed.unsqueeze(1)
        masked = masked.view(batch_size * self.num_sources, self.enc_dim, -1)  # B*S, F, M

        decoded = self.decoder(masked)  # B*S, 1, L'
        output = decoded.view(batch_size, self.num_sources, num_padded_frames)  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        
        if self.use_consistency:
            output = self._mixture_consistency_projection(output, x, mix_weights)

        return output
        

  
