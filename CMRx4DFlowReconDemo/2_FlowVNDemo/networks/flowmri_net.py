"""
Adopted from
https://github.com/cq615/Deep-MRI-Reconstruction/blob/master/cascadenet_pytorch/model_pytorch.py
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.misc_utils import *
import torch.utils.checkpoint as checkpoint


class FlowMRI_Net(nn.Module):
    def __init__(self, **kwargs):
        super(FlowMRI_Net, self).__init__()
        options = kwargs
        self.options = options

        self.nc = options['num_stages']  # number of iterations
        self.nf = options["features_out"]  # number of filters
        self.n_ch = options["features_in"]  # number of input channels
        self.gc = options["grad_check"]  # use gradient checkpointing 

        dc, wa = [], []
        for _ in range(self.nc):
            dc.append(dataConsistencyTerm(-2.2))   # sigmoid(-2.2)=0.1 for init
            wa.append(weightedAverageTerm(-2.2))
        self.dc = nn.ModuleList(dc)
        self.wa = nn.ModuleList(wa)
        
        self.bcrnn1 = BCRNNlayer(self.n_ch, self.nf, 3) 
        self.bcrnn2 = BCRNNlayer(self.nf, self.nf, 3)
        self.bcrnn3 = BCRNNlayer(self.nf, self.nf, 3)  
        self.bcrnn4 = BCRNNlayer(self.nf, self.nf, 3)  
        self.conv4_x = nn.Conv2d(self.nf, self.n_ch, 3, padding = 1, padding_mode = "reflect", dtype=torch.complex64)
        self.hid_init = Variable(torch.zeros((1,self.nf,1,1,1), dtype=torch.complex64)).cuda()

        self.complex_init()

    def complex_init(self):
        rng = np.random.RandomState(1337)
        for w in self.parameters():
            if w.data.dim() == 1 and w.data.dtype == torch.complex64:  # conv biases
                w.data *= 0
            if w.data.dim() == 4 and w.data.dtype == torch.complex64:  # conv weights
                w.data.real = torch.from_numpy(rng.normal(loc=0, scale=0.012, size=w.data.shape))
                w.data.imag = torch.from_numpy(rng.normal(loc=0, scale=0.012, size=w.data.shape))

    def run_denoise(self):
        def custom_forward(*inputs):
            hidden0_x = self.bcrnn1(inputs[0], inputs[1])  # combine/activate vertical (green) and horizontal (blue) passes  
            hidden1_x = self.bcrnn2(hidden0_x, inputs[2])  # combine/activate vertical (green) and horizontal (blue) passes 
            hidden2_x = self.bcrnn3(hidden1_x, inputs[3])  # combine/activate vertical (green) and horizontal (blue) passes 
            hidden3_x = self.bcrnn4(hidden2_x, inputs[4])  # combine/activate vertical (green) and horizontal (blue) passes 
            hidden4_x = self.conv4_x(hidden3_x.permute(0,2,1,3,4).view(-1, self.nf, self.width, self.height))   
            hidden4_x = hidden4_x.view(self.n_batch, self.n_seq, self.n_ch, self.width, self.height).permute(0,2,1,3,4) 
            return  hidden0_x, hidden1_x, hidden2_x, hidden3_x, hidden4_x
        return custom_forward
    
    def run_dc(self):
        def custom_forward(*inputs):
            return self.dc[inputs[3]].perform(inputs[0], inputs[1], inputs[2])
        return custom_forward
    
    def forward(self, x, k, c):
        self.n_batch, self.n_ch, self.n_seq, _, self.width, self.height = x.size()
        x = x[:,:,:,0]
        
        hid_init = self.hid_init.expand(self.n_batch, -1, self.n_seq, self.width, self.height)
        hidden = (hid_init, hid_init, hid_init, hid_init)

        for i in range(self.nc):
            # 5-layer CNN w/ 4 hidden states                 
            if self.gc:
                out = checkpoint.checkpoint(self.run_denoise(), x, hidden[0], hidden[1], hidden[2], hidden[3], use_reentrant=True)
            else:
                custom_forward = self.run_denoise()
                out = custom_forward(x, hidden[0], hidden[1], hidden[2], hidden[3])
            hidden = (out[0],out[1], out[2], out[3]) 
            x_cnn = out[4]

            # data-consistency
            if self.gc:
                Sx = checkpoint.checkpoint(self.run_dc(), x, torch.swapaxes(k, 1, 2), c, i, use_reentrant=True)[:,:,:,0]
            else:
                custom_forward3 = self.run_dc()
                Sx = custom_forward3(x, torch.swapaxes(k, 1, 2), c, i)[:,:,:,0]

            # weighted averaging
            x = self.wa[i].perform(x + x_cnn, Sx)

        return x[:,:,:,None]
    
class dataConsistencyTerm(nn.Module):

    def __init__(self, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(data=torch.Tensor([noise_lvl]))

    def perform(self, x, k0, sensitivity):
        """
        x    - input in image space (N V T H W)
        k0   - initially sampled elements in k-space (N V C T D H W)
        sensitivity - coil sensitivities (N C D H W)
        """
        x = sensitivity[:, :, None, None] * x[:, None, :, :, None]
        k = fftc2d(x) # convert into k-space 

        if self.noise_lvl is not None: # noisy case
            v = torch.sigmoid(self.noise_lvl) 
            out = torch.where(k0 != 0, v * k + (1 - v) * k0, k)
        else:  # noiseless case
            out = torch.where(k0 != 0, k0, k)
    
        x = ifftc2d(out)  # convert into image domain
        
        Sx = torch.sum(x * sensitivity.conj()[:, :, None, None], axis=1)  # coil combine
       
        return Sx


class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx):
        para = torch.sigmoid(self.para)  
        x = para*cnn + (1 - para)*Sx
        return x
    

class BCRNNlayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.CRNN_model = CRNNcell(input_size, hidden_size, kernel_size)
        self.hid_init = Variable(torch.zeros((1,hidden_size,1,1), dtype=torch.complex64)).cuda()
        
    def forward(self, input, input_iteration):
        """
        input - input image (batch_size, channel, num_seqs, width, height)
        input_iteration - hidden states from previous iteration (n_batch, hidden_size, n_seq, width, height)
        """
        nb, nc, nt, nx, ny = input.shape
        output_f, output_b = [], []

        # forward
        hidden = self.hid_init.expand(nb, -1, nx, ny)
        for i in range(nt): 
            hidden = self.CRNN_model(input[:,:,i], input_iteration[:,:,i], hidden)  # inputs: green, blue, orange
            output_f.append(hidden)

        # backward
        hidden = self.hid_init.expand(nb, -1, nx, ny)
        for i in range(nt)[::-1]:  
            hidden = self.CRNN_model(input[:,:,i], input_iteration[:,:,i], hidden)  # inputs: green (reversed), blue (reversed), orange
            output_b.append(hidden)

        output_f = torch.stack(output_f, dim=2)
        output_b = torch.stack(output_b[::-1], dim=2)
        output = output_f + output_b

        return output


class CRNNcell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2, padding_mode = "reflect", dtype=torch.complex64)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2, padding_mode = "reflect", dtype=torch.complex64)
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2, padding_mode = "reflect", dtype=torch.complex64)
        self.act = modReLU(hidden_size)

    def forward(self, input, hidden_iteration, hidden):
        """
        input - input image (batch_size, channel, width, height)
        hidden - hidden states in temporal dimension (batch_size, hidden_size, width, height)
        hidden_iteration - hidden states in iteration dimension (batch_size, hidden_size, width, height)
        """
        in_to_hid = self.i2h(input)  # green
        hid_to_hid = self.h2h(hidden)  # orange
        ih_to_ih = self.ih2ih(hidden_iteration)  # blue

        hidden = self.act(in_to_hid + hid_to_hid + ih_to_ih)  # combine/activate for horizontal (orange) pass

        return hidden
