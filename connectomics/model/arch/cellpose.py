import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage.filters import maximum_filter1d
from scipy.ndimage import binary_fill_holes, find_objects, mean

import fastremap
import cv2

from ..block import norm_act_conv2d


class Cellpose(nn.Module):
    
    def __init__(self,  **kwargs):
        super(Cellpose, self).__init__( **kwargs)
        self.model = CellposeModel( in_channel = 1, out_channel = 3, filters = [32, 64, 128, 256], 
                                    is_isotropic = False, isotropy = [False, False, False, True, True],
                                    pad_mode = 'replicate', act_mode = 'relu', norm_mode = 'bn',)
        self.dynamics = Dynamics()
    
    def run_dynamics(self, x):
        pass

    def forward(self, x):

        # 1. pass through the model
        # 2. run dynamics
        model_out = self.model(x)
        dynamics_out = self.run_dynamics(model_out)

        return dynamics_out


class CellposeModel(nn.Module):
    """Cellpose architecture

    Args:
        block_type (str): the block type at each U-Net stage. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each U-Net stage. Default: [28, 36, 48, 64, 80]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'relu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'bn'``
        init_mode (str): one of ``'xavier'``, ``'kaiming'``, ``'selu'`` or ``'orthogonal'``. Default: ``'orthogonal'``
        pooling (bool): downsample by max-pooling if `True` else using stride. Default: `False`
    """
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 filters: List[int] = [32, 64, 128, 256],
                 n_convs: int = 4,
                 kernel_size = (3,3), 
                 dilation = (1,1),
                 stride: int = 1,
                 groups: int = 1,
                 padding = (1,1),
                 pad_mode: str = 'replicate',
                 norm_mode: str = 'bn',
                 act_mode: str = 'relu',
                 **kwargs):
        super().__init__()
        
        self.nbase = filters
        self.downsample = DownsampleCP( filters, kernel_size = kernel_size, dilation = dilation,
                        n_convs = n_convs, stride = stride, groups = groups, padding = padding,
                        pad_mode = pad_mode, norm_mode = norm_mode, act_mode = act_mode)

        self.style = Style()

        self.upsample = UpsampleCP( filters, kernel_size = kernel_size, dilation = dilation,
                        n_convs = n_convs, stride = stride, groups = groups, padding = padding,
                        pad_mode = pad_mode, norm_mode = norm_mode, act_mode = act_mode )

        self.output = norm_act_conv2d( filters[0], out_channels, kernel_size = kernel_size, 
                        dilation = dilation, stride = stride, groups = groups, padding = padding,
                        pad_mode = pad_mode, norm_mode = norm_mode, act_mode = act_mode )

    def forward(self, x):

        downscaled_out = self.downsample(x)
        style = self.style(downscaled_out[-1])
        upscaled_out = self.upsample(style, downscaled_out)
        out = self.output(upscaled_out)

        return out


class Style(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):

        style = nn.AvgPool2d(kernel_size=(x.shape[-2],x.shape[-1]))(x)
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5
        
        return style


class StyleConv(nn.Module):

    def __init__(self, in_channels, out_channels, n_style, kernel_size = (3,3), dilation = (1,1), stride = 1, 
                groups = 1, padding = (1,1), pad_mode = 'replicate', norm_mode = 'bn', act_mode = 'relu'):

        self.style_feat = nn.Linear(n_style,out_channels)
        self.body = norm_act_conv2d(in_channels, out_channels, kernel_size = kernel_size, 
                        dilation = dilation, stride = stride, groups = groups, padding = padding,
                        pad_mode = pad_mode, norm_mode = norm_mode, act_mode = act_mode)
    
    def forward(self, style, x ):

        feat = self.style_feat(style).view(1,1,-1)
        out = self.body(feat + x)

        return out


class DownResBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 n_convs: int = 4,
                 kernel_size = (3,3), 
                 dilation = (1,1),
                 stride: int = 1,
                 groups: int = 1,
                 padding = (1,1),
                 pad_mode: str = 'replicate',
                 norm_mode: str = 'bn' , 
                 act_mode: str = 'relu'
                ):

        self.project = norm_act_conv2d(in_channels,out_channels,kernel_size=(1,1), dilation = dilation,
                            stride = stride, groups = groups, padding = padding, pad_mode = pad_mode, 
                            norm_mode = norm_mode, act_mode = 'none')
        
        block1 = []
        block2 = []

        for _ in range(n_convs//2):
            block1 += norm_act_conv2d(in_channels, out_channels, kernel_size = kernel_size, 
                            dilation = dilation, stride = stride, groups = groups, padding = padding,
                            pad_mode = pad_mode, norm_mode = norm_mode, act_mode = act_mode,
                            return_list = True)

        self.block1 = nn.Sequential(*block1)
        
        for _ in range(n_convs//2):
            block2 += norm_act_conv2d(in_channels, out_channels, kernel_size = kernel_size, 
                            dilation = dilation, stride = stride, groups = groups, padding = padding,
                            pad_mode = pad_mode, norm_mode = norm_mode, act_mode = act_mode,
                            return_list = True)

        self.block2 = nn.Sequential(*block2)

    def forward(self, x):

        block1_out = self.project(x) + self.block1(x)
        block2_out = block1_out + self.block2(block1_out)
        
        return block2_out


class DownsampleCP(nn.Module):
    def __init__(self,
                 n_convs: int = 4,
                 filters: List[int] = [32, 64, 128, 256],
                 kernel_size = (3,3), 
                 dilation = (1,1),
                 stride: int = 1,
                 groups: int = 1,
                 padding = (1,1),
                 pad_mode: str = 'replicate',
                 norm_mode: str = 'bn', 
                 act_mode: str = 'relu',
                 **kwargs
                ):

        super().__init__()
        
        modules = []
        for scale in range(len(filters)):
            modules.append(DownResBlock(filters[scale], filters[scale], n_convs = n_convs,
                        kernel_size = kernel_size, dilation = dilation, stride = stride,
                        groups = groups, padding = padding, pad_mode = pad_mode, 
                        norm_mode = norm_mode, act_mode = act_mode), kwargs)
        modules.append(nn.MaxPool2d(2, 2))

        self.down = nn.Sequential(modules)

    def forward(self, x):

        out_layers = []
        for i in len(self.down):
            if i==0:
                out_layers.append(x)
            else:
                out = self.down[i](x)
                out_layers.append(out)

        return out_layers


class UpResBlocks(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 n_convs: int = 4,
                 kernel_size = (3,3), 
                 dilation = (1,1),
                 stride: int = 1,
                 groups: int = 1,
                 padding = (1,1),
                 pad_mode: str = 'replicate',
                 norm_mode: str = 'bn', 
                 act_mode: str = 'relu'
                #  style_channels 
                ):
        super().__init__()
                
        self.project = norm_act_conv2d(in_channels,out_channels,kernel_size=(1,1), dilation = dilation,
                    stride = stride, groups = groups, padding = padding, pad_mode = pad_mode, 
                    norm_mode = norm_mode, act_mode = 'none')

        block1 = []
        block2 = []

        for i in range(n_convs//2):
            if i==0:
                block1 += norm_act_conv2d(in_channels, out_channels, kernel_size = kernel_size, 
                        dilation = dilation, stride = stride, groups = groups, padding = padding,
                        pad_mode = pad_mode, norm_mode = norm_mode, act_mode = act_mode,
                        return_list = True)
            else:
                block1 += norm_act_conv2d(in_channels, out_channels, kernel_size = kernel_size, 
                        dilation = dilation, stride = stride, groups = groups, padding = padding,
                        pad_mode = pad_mode, norm_mode = norm_mode, act_mode = act_mode,
                        return_list = True)

        self.block1 = nn.Sequential(*block1)
        
        for _ in range(n_convs//2):
            block2 += norm_act_conv2d(in_channels, out_channels, kernel_size = kernel_size, 
                    dilation = dilation, stride = stride, groups = groups, padding = padding,
                    pad_mode = pad_mode, norm_mode = norm_mode, act_mode = act_mode,
                    return_list = True)
        self.block2 = nn.Sequential(*block2)

    def forward(self, x, xd, style):

        x = self.project(x) + self.block1(style, x + xd )
        x = x + self.block2(style, x)
        
        return x


class UpsampleCP(nn.Module):
    def __init__(self,
                 filters: List[int] = [32, 64, 128, 256],
                 kernel_size = (3,3), 
                 dilation = (1,1),
                 stride: int = 1,
                 groups: int = 1,
                 padding=(1,1),
                 pad_mode: str = 'replicate',
                 act_mode: str = 'relu',
                 norm_mode: str = 'bn',
                 **kwargs
                 ):
        super().__init__()

        self.upsampling = nn.Upsample( scale_factor = 2, mode='nearest' )
        self.up_modules = nn.ModuleList()

        for scale in range(len(filters)):
            self.up_modules.append( UpResBlocks(filters[scale], filters[scale],
                            kernel_size=kernel_size, dilation=dilation,
                            stride=stride, groups=groups, padding=padding,
                            pad_mode=pad_mode, norm_mode=norm_mode, 
                            act_mode=act_mode), kwargs)

            self.up_modules.append(nn.MaxPool2d(2, 2))
        

    def forward(self, x, style):

        x = self.up_modules[-1](x[-1], x[-1], style)

        for n in range(len(self.up_modules)-2,-1,-1):
            out = self.upsampling(x)
            upsampled = self.up_modules[n](out,x[n], style)

        return upsampled
    

class Dynamics:
    
    def __init__(self):
        self.TORCH_ENABLED = True 
        self.torch_GPU = torch.device('cuda')
        self.torch_CPU = torch.device('cpu')
        try:
            from skimage import filters
            self.SKIMAGE_ENABLED = True
        except:
            self.SKIMAGE_ENABLED = False

    @njit(['(int16[:,:,:], float32[:], float32[:], float32[:,:])', 
            '(float32[:,:,:], float32[:], float32[:], float32[:,:])'], cache=True)
    def map_coordinates(self, I, yc, xc, Y):
        """
        bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y
        
        Parameters
        -------------
        I : C x Ly x Lx
        yc : ni
            new y coordinates
        xc : ni
            new x coordinates
        Y : C x ni
            I sampled at (yc,xc)
        """
        C,Ly,Lx = I.shape
        yc_floor = yc.astype(np.int32)
        xc_floor = xc.astype(np.int32)
        yc = yc - yc_floor
        xc = xc - xc_floor
        for i in range(yc_floor.shape[0]):
            yf = min(Ly-1, max(0, yc_floor[i]))
            xf = min(Lx-1, max(0, xc_floor[i]))
            yf1= min(Ly-1, yf+1)
            xf1= min(Lx-1, xf+1)
            y = yc[i]
            x = xc[i]
            for c in range(C):
                Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                        np.float32(I[c, yf, xf1]) * (1 - y) * x +
                        np.float32(I[c, yf1, xf]) * y * (1 - x) +
                        np.float32(I[c, yf1, xf1]) * y * x )


    def steps2D_interp(self, p, dP, niter, use_gpu=False, device=None, calc_trace=False):
        shape = dP.shape[1:]
        if self.device:
            if self.device is None:
                device = self.device
            shape = np.array(shape)[[1,0]].astype('double')-1  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
            pt = torch.from_numpy(p[[1,0]].T).double().to(device).unsqueeze(0).unsqueeze(0) # p is n_points by 2, so pt is [1 1 2 n_points]
            im = torch.from_numpy(dP[[1,0]]).double().to(device).unsqueeze(0) #covert flow numpy array to tensor on GPU, add dimension 
            # normalize pt between  0 and  1, normalize the flow
            for k in range(2): 
                im[:,k,:,:] *= 2./shape[k]
                pt[:,:,:,k] /= shape[k]
                
            # normalize to between -1 and 1
            pt = pt*2-1 
            
            # make an array to track the trajectories 
            if calc_trace:
                trace = torch.clone(pt).detach()
            
            #here is where the stepping happens
            for t in range(niter):
                if calc_trace:
                    trace = torch.cat((trace,pt))
                # align_corners default is False, just added to suppress warning
                dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
                
                for k in range(2): #clamp the final pixel locations
                    pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] + dPt[:,k,:,:], -1., 1.)
                

            #undo the normalization from before, reverse order of operations 
            pt = (pt+1)*0.5
            for k in range(2): 
                pt[:,:,:,k] *= shape[k]
                
            if calc_trace:
                trace = (trace+1)*0.5
                for k in range(2): 
                    trace[:,:,:,k] *= shape[k]
                    
            #pass back to cpu
            if calc_trace:
                tr =  trace[:,:,:,[1,0]].cpu().numpy().squeeze().T
            else:
                tr = None
            
            p =  pt[:,:,:,[1,0]].cpu().numpy().squeeze().T
            return p, tr
        else:
            dPt = np.zeros(p.shape, np.float32)
            if calc_trace:
                tr = np.zeros((p.shape[0],p.shape[1],niter))
            else:
                tr = None
                
            for t in range(niter):
                if calc_trace:
                    tr[:,:,t] = p.copy()
                self.map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)

                for k in range(len(p)):
                    p[k] = np.minimum(shape[k]-1, np.maximum(0, p[k] + dPt[k]))
            return p, tr

    @njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
    def steps3D(self, p, dP, inds, niter):
        """ run dynamics of pixels to recover masks in 3D
        
        Euler integration of dynamics dP for niter steps

        Parameters
        ----------------

        p: float32, 4D array
            pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)

        dP: float32, 4D array
            flows [axis x Lz x Ly x Lx]

        inds: int32, 2D array
            non-zero pixels to run dynamics on [npixels x 3]

        niter: int32
            number of iterations of dynamics to run

        Returns
        ---------------

        p: float32, 4D array
            final locations of each pixel after dynamics

        """
        shape = p.shape[1:]
        for t in range(niter):
            #pi = p.astype(np.int32)
            for j in range(inds.shape[0]):
                z = inds[j,0]
                y = inds[j,1]
                x = inds[j,2]
                p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
                p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] + dP[0,p0,p1,p2]))
                p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] + dP[1,p0,p1,p2]))
                p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] + dP[2,p0,p1,p2]))
        return p, None

    @njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32, boolean, boolean)', nogil=True)
    def steps2D(self, p, dP, inds, niter, calc_trace=False):
        """ run dynamics of pixels to recover masks in 2D
        
        Euler integration of dynamics dP for niter steps

        Parameters
        ----------------

        p: float32, 3D array
            pixel locations [axis x Ly x Lx] (start at initial meshgrid)

        dP: float32, 3D array
            flows [axis x Ly x Lx]

        inds: int32, 2D array
            non-zero pixels to run dynamics on [npixels x 2]

        niter: int32
            number of iterations of dynamics to run

        Returns
        ---------------

        p: float32, 3D array
            final locations of each pixel after dynamics

        """
        shape = p.shape[1:]
        if calc_trace:
            Ly = shape[0]
            Lx = shape[1]
            tr = np.zeros((niter,2,Ly,Lx))
        for t in range(niter):
            for j in range(inds.shape[0]):
                if calc_trace:
                    tr[t] = p.copy()
                # starting coordinates
                y = inds[j,0]
                x = inds[j,1]
                p0, p1 = int(p[0,y,x]), int(p[1,y,x])
                step = dP[:,p0,p1]

                for k in range(p.shape[0]):
                    p[k,y,x] = min(shape[k]-1, max(0, p[k,y,x] + step[k]))
        return p, tr


    def follow_flows(self, dP, mask=None, inds=None, niter=200, interp=True, use_gpu=True, device=None, calc_trace=False):
        """ define pixels and run dynamics to recover masks in 2D
        
        Pixels are meshgrid. Only pixels with non-zero cell-probability
        are used (as defined by inds)

        Parameters
        ----------------

        dP: float32, 3D or 4D array
            flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
        
        mask: (optional, default None)
            pixel mask to seed masks. Useful when flows have low magnitudes.

        niter: int (optional, default 200)
            number of iterations of dynamics to run

        interp: bool (optional, default True)
            interpolate during 2D dynamics (not available in 3D) 
            (in previous versions + paper it was False)

        use_gpu: bool (optional, default False)
            use GPU to run interpolated dynamics (faster than CPU)


        Returns
        ---------------

        p: float32, 3D array
            final locations of each pixel after dynamics

        """
        shape = np.array(dP.shape[1:]).astype(np.int32)
        niter = np.uint32(niter)
        if len(shape)>2:
            p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                    np.arange(shape[2]), indexing='ij')
            p = np.array(p).astype(np.float32)
            # run dynamics on subset of pixels
            #inds = np.array(np.nonzero(dP[0]!=0)).astype(np.int32).T
            inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
            p, tr = self.steps3D(p, dP, inds, niter)
        else:
            p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            # not sure why, but I had changed this to float64 at some point... tests showed that map_coordinates expects float32
            # possible issues elsewhere? 
            p = np.array(p).astype(np.float32)

            # added inds for debugging while preserving backwards compatibility 
            if inds is None:
                inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
            
            if inds.ndim < 2 or inds.shape[0] < 5:
                print('WARNING: no mask pixels found')
                return p, inds, None
            if not interp:
                p, tr = self.steps2D(p, dP.astype(np.float32), inds, niter, calc_trace=calc_trace)
                #p = p[:,inds[:,0], inds[:,1]]
                #tr = tr[:,:,inds[:,0], inds[:,1]].transpose((1,2,0))
            else:
                p_interp, tr = self.steps2D_interp(p[:,inds[:,0], inds[:,1]], dP, niter, use_gpu=use_gpu,
                                            device=device,   calc_trace=calc_trace)
                
                p[:,inds[:,0],inds[:,1]] = p_interp
        return p, inds, tr


    def remove_bad_flow_masks(self, masks, flows, threshold=0.4, use_gpu=False, device=None):
        """ remove masks which have inconsistent flows 
        
        Uses metrics.flow_error to compute flows from predicted masks 
        and compare flows to predicted flows from network. Discards 
        masks with flow errors greater than the threshold.

        Parameters
        ----------------

        masks: int, 2D or 3D array
            labelled masks, 0=NO masks; 1,2,...=mask labels,
            size [Ly x Lx] or [Lz x Ly x Lx]

        flows: float, 3D or 4D array
            flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

        threshold: float (optional, default 0.4)
            masks with flow error greater than threshold are discarded.

        Returns
        ---------------

        masks: int, 2D or 3D array
            masks with inconsistent flow masks removed, 
            0=NO masks; 1,2,...=mask labels,
            size [Ly x Lx] or [Lz x Ly x Lx]
        
        """
        merrors, _ = self.flow_error(masks, flows, use_gpu, device)
        badi = 1+(merrors>threshold).nonzero()[0]
        masks[np.isin(masks, badi)] = 0
        return masks


    def get_masks(self, p, iscell=None, rpad=20, flows=None, threshold=0.4, use_gpu=False, device=None):
        """ create masks using pixel convergence after running dynamics
        
        Makes a histogram of final pixel locations p, initializes masks 
        at peaks of histogram and extends the masks from the peaks so that
        they include all pixels with more than 2 final pixels p. Discards 
        masks with flow errors greater than the threshold. 
        Parameters
        ----------------
        p: float32, 3D or 4D array
            final locations of each pixel after dynamics,
            size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
        iscell: bool, 2D or 3D array
            if iscell is not None, set pixels that are 
            iscell False to stay in their original location.
        rpad: int (optional, default 20)
            histogram edge padding
        threshold: float (optional, default 0.4)
            masks with flow error greater than threshold are discarded 
            (if flows is not None)
        flows: float, 3D or 4D array (optional, default None)
            flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
            is not None, then masks with inconsistent flows are removed using 
            `remove_bad_flow_masks`.
        Returns
        ---------------
        M0: int, 2D or 3D array
            masks with inconsistent flow masks removed, 
            0=NO masks; 1,2,...=mask labels,
            size [Ly x Lx] or [Lz x Ly x Lx]
        
        """
        
        pflows = []
        edges = []
        shape0 = p.shape[1:]
        dims = len(p)
        if iscell is not None:
            if dims==3:
                inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                    np.arange(shape0[2]), indexing='ij')
            elif dims==2:
                inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                        indexing='ij')
            for i in range(dims):
                p[i, ~iscell] = inds[i][~iscell]

        for i in range(dims):
            pflows.append(p[i].flatten().astype('int32'))
            edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

        h,_ = np.histogramdd(tuple(pflows), bins=edges)
        hmax = h.copy()
        for i in range(dims):
            hmax = maximum_filter1d(hmax, 5, axis=i)

        seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
        Nmax = h[seeds]
        isort = np.argsort(Nmax)[::-1]
        for s in seeds:
            s = s[isort]

        pix = list(np.array(seeds).T)

        shape = h.shape
        if dims==3:
            expand = np.nonzero(np.ones((3,3,3)))
        else:
            expand = np.nonzero(np.ones((3,3)))
        for e in expand:
            e = np.expand_dims(e,1)

        for iter in range(5):
            for k in range(len(pix)):
                if iter==0:
                    pix[k] = list(pix[k])
                newpix = []
                iin = []
                for i,e in enumerate(expand):
                    epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                    epix = epix.flatten()
                    iin.append(np.logical_and(epix>=0, epix<shape[i]))
                    newpix.append(epix)
                iin = np.all(tuple(iin), axis=0)
                for p in newpix:
                    p = p[iin]
                newpix = tuple(newpix)
                igood = h[newpix]>2
                for i in range(dims):
                    pix[k][i] = newpix[i][igood]
                if iter==4:
                    pix[k] = tuple(pix[k])
        
        M = np.zeros(h.shape, np.uint32)
        for k in range(len(pix)):
            M[pix[k]] = 1+k
            
        for i in range(dims):
            pflows[i] = pflows[i] + rpad
        M0 = M[tuple(pflows)]

        # remove big masks
        uniq, counts = fastremap.unique(M0, return_counts=True)
        big = np.prod(shape0) * 0.4
        bigc = uniq[counts > big]
        if len(bigc) > 0 and (len(bigc)>1 or bigc[0]!=0):
            M0 = fastremap.mask(M0, bigc)
        fastremap.renumber(M0, in_place=True) #convenient to guarantee non-skipped labels
        M0 = np.reshape(M0, shape0)
        return M0


    def compute_masks(self, dP, cellprob, bd=None, p=None, inds=None, niter=200, mask_threshold=0.0, diam_threshold=12.,
                    flow_threshold=0.4, interp=True, do_3D=False, 
                    min_size=15, resize=None, verbose=False,
                    use_gpu=False,device=None,nclasses=3):
        """ compute masks using dynamics from dP, cellprob, and boundary """
        # if verbose:
        #     print('mask_threshold is %f',mask_threshold)
        
        cp_mask = cellprob > mask_threshold # analog to original iscell=(cellprob>cellprob_threshold)

        if np.any(cp_mask): #mask at this point is a cell cluster binary map, not labels     
            # follow flows
            if p is None:
                p , inds, tr = self.follow_flows(dP * cp_mask / 5., mask=cp_mask, inds=inds, niter=niter, interp=interp, 
                                                use_gpu=use_gpu, device=device)
                if inds.ndim < 2 or inds.shape[0] < 5:
                    print('No cell pixels found.')
                    shape = resize if resize is not None else cellprob.shape
                    mask = np.zeros(shape, np.uint16)
                    p = np.zeros((len(shape), *shape), np.uint16)
                    return mask, p, []
            else: 
                if verbose:
                    print('p given')
            
            #calculate masks
            mask = self.get_masks(p, iscell=cp_mask, flows=dP, use_gpu=use_gpu)
                
            # flow thresholding factored out of get_masks
            if not do_3D:
                shape0 = p.shape[1:]
                if mask.max()>0 and flow_threshold is not None and flow_threshold > 0:
                    # make sure labels are unique at output of get_masks
                    mask = self.remove_bad_flow_masks(mask, dP, threshold=flow_threshold, use_gpu=use_gpu, device=device)
            
            if resize is not None:
                #if verbose:
                #    print(f'resizing output with resize = {resize}')
                if mask.max() > 2**16-1:
                    recast = True
                    mask = mask.astype(np.float32)
                else:
                    recast = False
                    mask = mask.astype(np.uint16)
                mask = self.resize_image(mask, resize[0], resize[1], interpolation=cv2.INTER_NEAREST)
                if recast:
                    mask = mask.astype(np.uint32)
                Ly,Lx = mask.shape
            elif mask.max() < 2**16:
                mask = mask.astype(np.uint16)

        else: # nothing to compute, just make it compatible
            print('No cell pixels found.')
            shape = resize if resize is not None else cellprob.shape
            mask = np.zeros(shape, np.uint16)
            p = np.zeros((len(shape), *shape), np.uint16)
            return mask, p, []


        # moving the cleanup to the end helps avoid some bugs arising from scaling...
        # maybe better would be to rescale the min_size and hole_size parameters to do the
        # cleanup at the prediction scale, or switch depending on which one is bigger... 
        mask = self.fill_holes_and_remove_small_masks(mask, min_size=min_size)

        return mask, p, []


    def flow_error(self, maski, dP_net, use_gpu=False, device=None):
        """ error in flows from predicted masks vs flows predicted by network run on image

        This function serves to benchmark the quality of masks, it works as follows
        1. The predicted masks are used to create a flow diagram
        2. The mask-flows are compared to the flows that the network predicted

        If there is a discrepancy between the flows, it suggests that the mask is incorrect.
        Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
        changed in Cellpose.eval or CellposeModel.eval.

        Parameters
        ------------
        
        maski: ND-array (int) 
            masks produced from running dynamics on dP_net, 
            where 0=NO masks; 1,2... are mask labels
        dP_net: ND-array (float) 
            ND flows where dP_net.shape[1:] = maski.shape

        Returns
        ------------

        flow_errors: float array with length maski.max()
            mean squared error between predicted flows and flows from masks
        dP_masks: ND-array (float)
            ND flows produced from the predicted masks
        
        """
        if dP_net.shape[1:] != maski.shape:
            print('ERROR: net flow is not same size as predicted masks')
            return

        # flows predicted from estimated masks
        dP_masks = self.masks_to_flows(maski, use_gpu=use_gpu, device=device)
        # difference between predicted flows vs mask flows
        flow_errors=np.zeros(maski.max())
        for i in range(dP_masks.shape[0]):
            flow_errors += mean((dP_masks[i] - dP_net[i]/5.)**2, maski,
                                index=np.arange(1, maski.max()+1))

        return flow_errors, dP_masks


    
    def resize_image(self, img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR, no_channels=False):
        """ resize image for computing flows / unresize for computing dynamics

        Parameters
        -------------

        img0: ND-array
            image of size [Y x X x nchan] or [Lz x Y x X x nchan] or [Lz x Y x X]

        Ly: int, optional

        Lx: int, optional

        rsz: float, optional
            resize coefficient(s) for image; if Ly is None then rsz is used

        interpolation: cv2 interp method (optional, default cv2.INTER_LINEAR)

        Returns
        --------------

        imgs: ND-array 
            image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        """
        if Ly is None and rsz is None:
            error_message = 'must give size to resize to or factor to use for resizing'
            print(error_message)
            raise ValueError(error_message)

        if Ly is None:
            # determine Ly and Lx using rsz
            if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
                rsz = [rsz, rsz]
            if no_channels:
                Ly = int(img0.shape[-2] * rsz[-2])
                Lx = int(img0.shape[-1] * rsz[-1])
            else:
                Ly = int(img0.shape[-3] * rsz[-2])
                Lx = int(img0.shape[-2] * rsz[-1])
        
        # no_channels useful for z-stacks, sot he third dimension is not treated as a channel
        # but if this is called for grayscale images, they first become [Ly,Lx,2] so ndim=3 but 
        if (img0.ndim>2 and no_channels) or (img0.ndim==4 and not no_channels):
            if no_channels:
                imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
            else:
                imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
            for i,img in enumerate(img0):
                imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
        else:
            imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
        return imgs


    # Edited slightly to only remove small holes(under min_size) to avoid filling in voids formed by cells touching themselves
    # (Masks show this, outlines somehow do not. Also need to find a way to split self-contact points).
    def fill_holes_and_remove_small_masks(self, masks, min_size=15, hole_size=3, scale_factor=1):
        """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
        
        fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
        
        Parameters
        ----------------

        masks: int, 2D or 3D array
            labelled masks, 0=NO masks; 1,2,...=mask labels,
            size [Ly x Lx] or [Lz x Ly x Lx]

        min_size: int (optional, default 15)
            minimum number of pixels per mask, can turn off with -1

        Returns
        ---------------

        masks: int, 2D or 3D array
            masks with holes filled and masks smaller than min_size removed, 
            0=NO masks; 1,2,...=mask labels,
            size [Ly x Lx] or [Lz x Ly x Lx]
        
        """
        # * Changed *
            
        hole_size *= scale_factor
            
        if masks.ndim > 3 or masks.ndim < 2:
            raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
        
        slices = find_objects(masks)
        j = 0
        for i,slc in enumerate(slices):
            if slc is not None:
                msk = masks[slc] == (i+1)
                npix = msk.sum()
                if min_size > 0 and npix < min_size:
                    masks[slc][msk] = 0
                elif npix > 0:   
                    if msk.ndim==3:
                        for k in range(msk.shape[0]):
                            # Omnipose version (breaks 3D tests)
                            # padmsk = remove_small_holes(np.pad(msk[k],1,mode='constant'),hsz)
                            # msk[k] = padmsk[1:-1,1:-1]
                            
                            #Cellpose version
                            msk[k] = binary_fill_holes(msk[k])

                    else:          
                        if self.SKIMAGE_ENABLED: # Omnipose version (passes 2D tests)
                            hsz = np.count_nonzero(msk)*hole_size/100 #turn hole size into percentage
                            padmsk = self.remove_small_holes(np.pad(msk,1,mode='constant'),hsz)
                            msk = padmsk[1:-1,1:-1]
                        else: #Cellpose version
                            msk = binary_fill_holes(msk)
                    masks[slc][msk] = (j+1)
                    j+=1
        return masks
