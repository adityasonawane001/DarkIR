import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

try:
    from .arch_util import LayerNorm2d
except:
    from arch_util import LayerNorm2d


class SimpleGate(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        # If channels is provided, use Gated Attention (Change 4)
        if channels is not None:
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels // 2, channels // 8),
                nn.ReLU(),
                nn.Linear(channels // 8, channels // 2),
                nn.Sigmoid()
            )
        else:
            self.channel_attn = None

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        if self.channel_attn is not None:
            attn = self.channel_attn(x2)
            attn = attn.view(attn.shape[0], -1, 1, 1)
            return x1 * (x2 * attn)
        return x1 * x2

class Adapter(nn.Module):
    
    def __init__(self, c, ffn_channel = None):
        super().__init__()
        if ffn_channel:
            ffn_channel = 2
        else:
            ffn_channel = c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.depthwise = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1)

    def forward(self, input):
        
        x = self.conv1(input) + self.depthwise(input)
        x = self.conv2(x)
        
        return x

class FreMLP(nn.Module):
    
    def __init__(self, nc, expand = 2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))
        
        # Change 1: Phase Attention Layer
        self.phase_conv = nn.Conv2d(nc, nc, 1, 1, 0)

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x.float(), norm='backward')
        
        # Robust amplitude and phase calculation to prevent NaN gradients!
        real = x_freq.real
        imag = x_freq.imag
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        pha = torch.atan2(imag, real + 1e-8)
        
        # Enhance amplitude
        mag = self.process1(mag)
        mag = torch.nan_to_num(mag, nan=0.0, posinf=1e4, neginf=0.0)
        mag = torch.clamp(mag, min=0.0, max=1e4)
        
        # Change 1: Enhance phase with attention
        phase_weight = torch.sigmoid(self.phase_conv(pha))
        phase_weight = torch.nan_to_num(phase_weight, nan=0.5, posinf=1.0, neginf=0.0)
        pha = pha * phase_weight
        
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out = torch.nan_to_num(x_out, nan=0.0, posinf=1e4, neginf=-1e4)
        return x_out

class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''
    def __init__(self, c, DW_Expand, dilation = 1):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        
        self.branch = nn.Sequential(
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                            bias=True, dilation = dilation) # the dconv
        )
    def forward(self, input):
        return self.branch(input)
    
class DBlock(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand = 1, dilation = dilation))
            
        # Change 2: Adaptive Dilation Rates
        self.rate_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.dw_channel, len(dilations)),
            nn.Softmax(dim=1)
        )
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate(self.dw_channel) # Change 4: Gated Attention
        self.sg2 = SimpleGate(FFN_Expand * c) # Change 4: Gated Attention
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


#        self.adapter = Adapter(c, ffn_channel=None)
        
#        self.use_adapters = False

#    def set_use_adapters(self, use_adapters):
#        self.use_adapters = use_adapters
        
    def forward(self, inp, adapter = None):

        y = inp
        x = self.norm1(inp)
        # x = self.conv1(self.extra_conv(x))
        x = self.extra_conv(self.conv1(x))
        
        # Change 2: Apply adaptive dilation weights
        w = self.rate_weights(x)
        
        z = 0
        for i, branch in enumerate(self.branches):
            w_i = w[:, i:i+1, None, None]
            z += w_i * branch(x)
        
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        #second step
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]
        x = y + x * self.gamma
        
#        if self.use_adapters:
#            return self.adapter(x)
#        else:
        return x 

class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c, DW_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
                
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation))
            
        # Change 2: Adaptive Dilation Rates
        self.rate_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.dw_channel, len(dilations)),
            nn.Softmax(dim=1)
        )
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate(self.dw_channel) # Change 4: Gated Attention
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc = c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


#        self.adapter = Adapter(c, ffn_channel=None)
        
#        self.use_adapters = False

#    def set_use_adapters(self, use_adapters):
#        self.use_adapters = use_adapters

    def forward(self, inp):
        y = inp
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        
        # Change 2: Apply adaptive dilation weights
        w = self.rate_weights(x)
        
        z = 0
        for i, branch in enumerate(self.branches):
            w_i = w[:, i:i+1, None, None]
            z += w_i * branch(x)
        
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        #second step
        x_step2 = self.norm2(y) # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2) # size [B, C, H, W]
        x = y * x_freq 
        x = y + x * self.gamma

#        if self.use_adapters:
#            return self.adapter(x)
#        else:
        return x 

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    img_channel = 3
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num = 3
    dec_blks = [3, 1, 1]
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    # net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    net  = EBlock(c = img_channel, 
                            dilations = dilations,
                            extra_depth_wise=extra_depth_wise)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    output = net(torch.randn((4, 3, 256, 256)))
    # print('Values of EBlock:')
    print(macs, params)

    channels = 128
    resol = 32
    ksize = 5

    # net = FAC(channels=channels, ksize=ksize)
    # inp_shape = (channels, resol, resol)
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    # print('Values of FAC:')
    # print(macs, params)
