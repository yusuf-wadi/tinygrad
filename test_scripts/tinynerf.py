from tinygrad import Device
print(Device.DEFAULT)
import torch #TODO replace torch where you can
import numpy as np
from tqdm import tqdm
import tinygrad.nn as nn
from tinygrad.tensor import Tensor, dtypes
from tinygrad.function import Relu, Sigmoid
from tinygrad import TinyJit
from PIL import Image
from typing import List, Dict, Optional
import os
import requests
import matplotlib.pyplot as plt

class tinyNGP:
    
    def __init__(self, T, Nl, L, device, aabb_scale, F=2):
        self.T = T # num entries in hash table
        self.Nl = Nl # list of resolutions
        self.F = F # dimension of each entry in hash table
        self.L = L # for encoding directions of light into high-dimensional space
        self.device = device
        self.aabb_scale = aabb_scale # bodied volume scale
        
        self.lookup_tables: Dict[str, Tensor] ={
                str(i): (Tensor.rand((T,F), device=device) * 2 - 1 * 1e-4) # a grid of T x F 
                for i in range(len(Nl)) # for each resolution in Nl
                }
        self.pi1, self.pi2, self.pi3 = 1, 2_654_435_761, 805_459_861 # three primes for hashing
        
        self.density_MLP_units_out = 16
        self.dir3d_uncode = 27 # 3^3
        
        self.density_MLP: List =[
            nn.Linear(self.F * len(Nl), 64),
            Relu(device=self.device).apply, # not sure how to contextualize device here
            nn.Linear(64, self.density_MLP_units_out)
        ]
        
        self.color_MLP: List =[
            nn.Linear(self.dir3d_uncode + self.density_MLP_units_out, 64),
            Relu(device=self.device).apply,
            nn.Linear(64, 64),
            Relu(device=self.device).apply,
            nn.Linear(64, 3),
            Sigmoid(device=self.device).apply 
        ]

        
    def positional_encoding(self, x: Tensor):
        out = [x]
        for j in range(self.L):
            out.append((2**j * x).sin())
            out.append((2**j * x).cos())
        #print(out)
        return Tensor.cat(*out, dim=1)

    def __call__(self, x: Tensor, d)-> tuple[Tensor, Tensor]:
        x = x / self.aabb_scale
        assert x.shape[1] == 3, "Input tensor x must have shape (N, 3)"
        mask = (x[:, 0].abs() < .5).int() & (x[:, 1].abs() < .5).int() & (x[:, 2].abs() < .5).int()
        mask.requires_grad = False
        mask = mask.to(x.device)
        x = x + 0.5  # x in [0, 1]^3
        
        color = Tensor.zeros((x.shape[0], 3), device=x.device).contiguous()
        log_sigma = (Tensor.zeros((x.shape[0],), device=x.device) - 1e5).contiguous()
        log_sigma.requires_grad = False
        
        features = Tensor.zeros((x.shape[0], self.F * len(self.Nl)), device=x.device).contiguous()
        
        for i, N in enumerate(self.Nl):
            floor = Tensor.floor(x * N)
            ceil = Tensor.ceil(x * N)
            vertices = Tensor.zeros((x[mask].shape[0], 8, 3), dtype=dtypes.int32, device=x.device)
            vertices = vertices.contiguous()
            vertices[:, 0] = floor
            vertices[:, 1] = Tensor.cat(ceil[:, 0, None], floor[:, 1, None], floor[:, 2, None], dim=1)
            vertices[:, 2] = Tensor.cat(floor[:, 0, None], ceil[:, 1, None], floor[:, 2, None], dim=1)
            vertices[:, 3] = Tensor.cat(ceil[:, 0, None], ceil[:, 1, None], floor[:, 2, None], dim=1)
            vertices[:, 4] = Tensor.cat(floor[:, 0, None], floor[:, 1, None], ceil[:, 2, None], dim=1)
            vertices[:, 5] = Tensor.cat(ceil[:, 0, None], floor[:, 1, None], ceil[:, 2, None], dim=1)
            vertices[:, 6] = Tensor.cat(floor[:, 0, None], ceil[:, 1, None], ceil[:, 2, None], dim=1)
            vertices[:, 7] = ceil
            
            # hashing
            a = vertices[:, :, 0] * self.pi1
            b = vertices[:, :, 1] * self.pi2
            c = vertices[:, :, 2] * self.pi3
            H_x = Tensor.xor(Tensor.xor(a, b), c).remainder(self.T).int()
        
            # lookup
            looked_up = self.lookup_tables[str(i)][H_x].transpose(-1, -2)
            volume = looked_up.reshape((looked_up.shape[0], 2, 2, 2, 2))
            # Debugging print statements to verify shapes
            # print(f"volume shape: {volume.shape}")
            # print(f"grid shape: {((x[mask] * N - floor) - 0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1).shape}")
            # #print(f"grid_sample_5d output shape: {grid_sample_5d(volume, ((x[mask] * N - floor) - 0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1)).squeeze(-1).reshape(1, self.F).shape}")
            # print(f"features shape: {features.shape}")

            features[:, i*2:(i+1)*2] = Tensor(torch.nn.functional.grid_sample(
                torch.tensor(volume.numpy(), dtype=torch.float32),
                torch.tensor(((x[mask] * N - floor) - 0.5).numpy(), dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                ).squeeze(-1).squeeze(-1).squeeze(-1).numpy())# this is still bad, but it's a start
               
        xi = self.positional_encoding(d)
        h = features.sequential(self.density_MLP)
        #h.requires_grad = False
        #print(f"xi shape: {xi.shape}")
        #print(f"h[:, 1:] shape: {h[:, 1:].shape}")
        log_sigma = Tensor.where(mask, h[:, 0], log_sigma)
        color = Tensor.where(mask.unsqueeze(-1), h.cat(xi, dim=1).sequential(self.color_MLP), color)

        return color, Tensor.exp(log_sigma)

class TinyNerf:
    """
    a very tiny nerf model. implementation of a tiny neural radiance field (now tinier in tinygrad!): https://arxiv.org/abs/2003.08934
    """
    def __init__(self, filter_size=128, num_encoding_functions=6):
        super().__init__() # https://stackoverflow.com/questions/576169/understanding-python-super-and-init-methods
        # Input layer (default: 39 -> 128)
        self.layer1 = nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = Relu.apply
        
    def __call__(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
        
    
     
## tiny nerf utilitites ##
def meshgrid(x, y):
  """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
        tensor1 (Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
        tensor2 (Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
  """
  grid_x = Tensor.cat(*[x[idx:idx+1].expand(y.shape).unsqueeze(0) for idx in range(x.shape[0])])
  grid_y = Tensor.cat(*[y.unsqueeze(0)]*x.shape[0])
  return grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)

def roll(tensor: Tensor, shifts, dims=None):
    if dims is None:
        original_shape = tensor.shape
        tensor = tensor.flatten()
        dims = (0,)
        shifts = (shifts,)
    
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)
    
    assert len(shifts) == len(dims), "shifts and dims must have the same length"
    
    rolled_tensor = tensor
    for shift, dim in zip(shifts, dims):
        shift = shift % rolled_tensor.shape[dim]  # Effective shift
        if shift == 0:
            continue
        
        # Perform the roll using pad and shrink
        pad_width = [(0, 0)] * rolled_tensor.ndim
        pad_width[dim] = (0, shift)
        padded = rolled_tensor.pad(pad_width)
        
        shrink_indices = [(0, s) for s in rolled_tensor.shape]
        shrink_indices[dim] = (shift, shift + rolled_tensor.shape[dim])
        rolled_tensor = padded.shrink(shrink_indices)
    
    if len(dims) == 1 and dims[0] == 0:
        rolled_tensor = rolled_tensor.reshape(original_shape)
    
    return rolled_tensor

def cumprod_exclusive(tensor: Tensor) -> Tensor:
  r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in tinygrad.

  Args:
    tensor (Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
  
  Returns:
    cumprod (Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
  """
  # TESTED
  # Only works for the last dimension (dim=-1)
  dim = -1
  # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
  cprod = Tensor(np.cumprod(tensor.numpy(), dim))
  # "Roll" the elements along dimension 'dim' by 1 element.
  cprod = roll(cprod, 1, dim).contiguous()
  # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
  cprod[..., 0] = 1.
  
  return cprod

def get_ray_bundle(height: int, width: int, focal_length: float, tform_cam2world: Tensor):
  r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

  Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.
  
  Returns:
    ray_origins (Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
  """
  # TESTED
  ii, jj = meshgrid(
      Tensor.arange(width),
      Tensor.arange(height)
  )
  directions = Tensor.stack((ii - width * .5) / focal_length,
                            -(jj - height * .5) / focal_length,
                            -Tensor.ones_like(ii)
                           , dim=-1)
  ray_directions = Tensor.sum(directions[..., None, :] * tform_cam2world[:3, :3], axis=-1)
  ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
  return ray_origins, ray_directions

def compute_query_points_from_rays(
    ray_origins: Tensor,
    ray_directions: Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize: Optional[bool] = True
) -> tuple[Tensor, Tensor]:
  r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
  variables indicate the bounds within which 3D points are to be sampled.

  Args:
    ray_origins (Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    ray_directions (Tensor): Direction of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
      coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
      coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
      randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
      By default, this is set to `True`. If disabled (by setting to `False`), we sample
      uniformly spaced points along each ray in the "bundle".
  
  Returns:
    query_points (Tensor): Query points along each ray
      (shape: :math:`(width, height, num_samples, 3)`).
    depth_values (Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
  """
  # TESTED
  # shape: (num_samples)
  depth_values = Tensor(np.linspace(near_thresh, far_thresh, num_samples)).to(ray_origins.device)
  if randomize is True:
    # ray_origins: (width, height, 3)
    # noise_shape = (width, height, num_samples)
    noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
    # depth_values: (num_samples)
    depth_values = depth_values \
        + Tensor.rand(noise_shape).to(ray_origins.device) * (far_thresh
            - near_thresh) / num_samples
  # (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
  # query_points:  (width, height, num_samples, 3)
  query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
  # TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
  return query_points, depth_values

def render_volume_density(
    radiance_field: Tensor,
    ray_origins: Tensor,
    depth_values: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
  r"""Differentiably renders a radiance field, given the origin of each ray in the
  "bundle", and the sampled depth values along them.

  Args:
    radiance_field (Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
  
  Returns:
    rgb_map (Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
  """
  # TESTED
  sigma_a = Relu.apply(radiance_field[..., 3])
  rgb = Sigmoid.apply(radiance_field[..., :3])
  one_e_10 = Tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
  dists = Tensor.cat(depth_values[..., 1:] - depth_values[..., :-1],
                  one_e_10.expand(depth_values[..., :1].shape), dim=-1)
  alpha = 1. - Tensor.exp(-sigma_a * dists)
  weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

  rgb_map = (weights[..., None] * rgb).sum(axis=-2)
  depth_map = (weights * depth_values).sum(axis=-1)
  acc_map = weights.sum(-1)

  return rgb_map, depth_map, acc_map

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> Tensor:
  r"""Apply positional encoding to the input.

  Args:
    tensor (Tensor): Input tensor to be positionally encoded.
    num_encoding_functions (optional, int): Number of encoding functions used to
        compute a positional encoding (default: 6).
    include_input (optional, bool): Whether or not to include the input in the
        computed positional encoding (default: True).
    log_sampling (optional, bool): Sample logarithmically in frequency space, as
        opposed to linearly (default: True).
  
  Returns:
    (Tensor): Positional encoding of the input tensor.
  """
  # TESTED
  # Trivially, the input tensor is added to the positional encoding.
  encoding = [tensor] if include_input else []
  # Now, encode the input using a set of high-frequency functions and append the
  # resulting values to the encoding.
  frequency_bands = None
  if log_sampling:
      frequency_bands = Tensor(2.0 ** np.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            #dtype=tensor.dtype,
        )).to(tensor.device)
  else:
      frequency_bands = Tensor(np.linspace(
          2.0 ** 0.0,
          2.0 ** (num_encoding_functions - 1),
          num_encoding_functions,
          #dtype=tensor.dtype,
      )).to(tensor.device)

  for freq in frequency_bands:
      for func in [Tensor.sin, Tensor.cos]:
          encoding.append(func(tensor * freq))

  # Special case, for no positional encoding
  if len(encoding) == 1:
      return encoding[0]
  else:
      return Tensor.cat(*encoding, dim=-1)
  

def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
  r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
  Each element of the list (except possibly the last) has dimension `0` of length
  `chunksize`.
  """
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

class TinyDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.batch_idx = 0

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.batch_idx:self.batch_idx + self.batch_size]
        batch_data = self.dataset[batch_indices.tolist()]
        self.batch_idx += self.batch_size
        return batch_data

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def test(model, hn, hf, dataset, img_index, chunk_size=20, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    px_values = []   # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        px_values.append(render_rays(model, ray_origins_, ray_directions_,
                                     hn=hn, hf=hf, nb_bins=nb_bins))
    img = Tensor.cat(*px_values, dim=0).numpy().reshape(H, W, 3)
    img = (img.clip(0, 1)*255.).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f'novel_views/img_{img_index}.png')
    
def _test_deb(model, hn, hf, dataset, img_index, chunk_size=20, nb_bins=192, H=25, W=40):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    
    print(f"Ray origins shape: {ray_origins.shape}")
    print(f"Ray directions shape: {ray_directions.shape}")
    
    px_values = []
    for i in range(int(np.ceil(H * W / chunk_size))):  # Changed to iterate over all pixels
        start = i * chunk_size
        end = min((i + 1) * chunk_size, H * W)
        ray_origins_ = ray_origins[start:end].to(device)
        ray_directions_ = ray_directions[start:end].to(device)
        chunk_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        px_values.append(chunk_values)
        print(f"Chunk {i} shape: {chunk_values.shape}")
    print(f"Number of chunks: {len(px_values)}")
    if px_values:
        print(f"Shape of first chunk: {px_values[0].shape}")
    if not px_values:
        print("px_values is empty!")
        return
    
    img = Tensor.cat(*px_values, dim=0)
    img = tensor_to_img(img, H, W)
    img.save(f'novel_views/img_{img_index}.png')

def tensor_to_img(tensor, H, W):
    """
    Convert a tensor to a PIL image.
    """
    img = tensor.numpy().reshape(H, W, 3)
    img = (img.clip(0, 1)*255.).astype(np.uint8)
    img = Image.fromarray(img)
    return img

#TODO: fix in-house grid_sample_5d
def grid_sample_5d(input: Tensor, grid: Tensor, mode='bilinear', padding_mode='zeros', align_corners=False):
    # Ensure input is a 5D tensor
    assert len(input.shape) == 5, "Input must be a 5D tensor"
    # Ensure grid is a 5D tensor
    assert len(grid.shape) == 5, "Grid must be a 5D tensor"
    assert grid.shape[-1] == 3, "Grid must have 3 channels (x, y, z)"

    # Extract dimensions
    N, C, D, H, W = input.shape
    _, out_D, out_H, out_W, _ = grid.shape

    # Normalize grid coordinates to [-1, 1]
    if align_corners:
        grid = ((grid + 1) / 2) * Tensor([D-1, H-1, W-1], device=grid.device)
    else:
        grid = ((grid + 1) * Tensor([D, H, W], device=grid.device) - 1) / 2

    # Clip coordinates to be within the volume bounds
    if padding_mode == 'zeros':
        grid = grid.clip(-1, 1)
    elif padding_mode == 'border':
        grid = grid.clip(0, Tensor([D-1, H-1, W-1], device=grid.device))
    elif padding_mode == 'reflection':
        grid = ((grid.abs() - Tensor([D, H, W], device=grid.device)).abs() - Tensor([D, H, W], device=grid.device)).abs()

    # Split grid into x, y, and z coordinates
    z, y, x = grid.split(1, dim=-1)

    # Compute interpolation weights
    z0 = z.floor().cast(dtypes.int32)
    z1 = z0 + 1
    y0 = y.floor().cast(dtypes.int32)
    y1 = y0 + 1
    x0 = x.floor().cast(dtypes.int32)
    x1 = x0 + 1

    # Clip indices to volume size
    z0 = z0.clip(0, D-1)
    z1 = z1.clip(0, D-1)
    y0 = y0.clip(0, H-1)
    y1 = y1.clip(0, H-1)
    x0 = x0.clip(0, W-1)
    x1 = x1.clip(0, W-1)

    # Compute interpolation weights
    wa = (z1.float() - z) * (y1.float() - y) * (x1.float() - x)
    wb = (z1.float() - z) * (y1.float() - y) * (x - x0.float())
    wc = (z1.float() - z) * (y - y0.float()) * (x1.float() - x)
    wd = (z1.float() - z) * (y - y0.float()) * (x - x0.float())
    we = (z - z0.float()) * (y1.float() - y) * (x1.float() - x)
    wf = (z - z0.float()) * (y1.float() - y) * (x - x0.float())
    wg = (z - z0.float()) * (y - y0.float()) * (x1.float() - x)
    wh = (z - z0.float()) * (y - y0.float()) * (x - x0.float())

    # Gather voxel values from input tensor
    Ia = input[:, :, z0.squeeze(-1), y0.squeeze(-1), x0.squeeze(-1)]
    Ib = input[:, :, z0.squeeze(-1), y0.squeeze(-1), x1.squeeze(-1)]
    Ic = input[:, :, z0.squeeze(-1), y1.squeeze(-1), x0.squeeze(-1)]
    Id = input[:, :, z0.squeeze(-1), y1.squeeze(-1), x1.squeeze(-1)]
    Ie = input[:, :, z1.squeeze(-1), y0.squeeze(-1), x0.squeeze(-1)]
    If = input[:, :, z1.squeeze(-1), y0.squeeze(-1), x1.squeeze(-1)]
    Ig = input[:, :, z1.squeeze(-1), y1.squeeze(-1), x0.squeeze(-1)]
    Ih = input[:, :, z1.squeeze(-1), y1.squeeze(-1), x1.squeeze(-1)]

    # Perform trilinear interpolation
    output = (Ia*wa + Ib*wb + Ic*wc + Id*wd + Ie*we + If*wf + Ig*wg + Ih*wh)

    # Reshape the output to match the expected shape
    N, C, D, H, W = input.shape
    _, out_D, out_H, out_W, _ = grid.shape
    return output.reshape(N, C, out_D, out_H, out_W)

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = Tensor(np.cumprod(alphas, dim=1))
    return Tensor.cat(Tensor.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1], dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, train=False, gt: Tensor=None, optim: nn.optim.LAMB=None, hn=0, hf=0.5, nb_bins=192)-> Tensor:
    device = ray_origins.device
    t = Tensor(np.linspace(hn, hf, nb_bins)).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = Tensor.cat(t[:, :1], mid, dim=-1)
    upper = Tensor.cat(mid, t[:, -1:], dim=-1)
    u = Tensor.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = Tensor.cat(t[:, 1:] - t[:, :-1], Tensor(
        [1e10], device=device).expand(ray_origins.shape[0], 1), dim=-1)
    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    alpha = 1 - Tensor.exp(-sigma.reshape(x.shape[:-1]) * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors.reshape(x.shape)).sum(axis=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    pred_px_values = c + 1 - weight_sum.unsqueeze(-1)
    
    if train:
        optim.zero_grad()
        loss = ((gt - pred_px_values) ** 2).mean().backward()
        for param in optim.params:
                    if param.grad is None:
                        param.grad = Tensor.zeros_like(param)
        optim.step()
        return loss, optim
    
    return pred_px_values

def render_rays_jit(nerf_model, ray_origins, ray_directions, train=False, gt: Tensor=None, optim: nn.optim.LAMB=None, hn=0, hf=0.5, nb_bins=192) -> Tensor:
    return render_rays(nerf_model, ray_origins, ray_directions, train, gt, optim, hn, hf, nb_bins)

@TinyJit
@Tensor.train()
def train(nerf_model: tinyNGP, optimizer: nn.optim.LAMB, data_loader: TinyDataLoader, device='cpu', hn=0, hf=1, nb_epochs=10, nb_bins=192, H=400, W=400):

    for _ in range(nb_epochs):
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            gt_px_values = batch[:, 6:].to(device)
            loss, optimizer = render_rays_jit(nerf_model, ray_origins, ray_directions, train=True, gt=gt_px_values, optim=optimizer, hn=hn, hf=hf, nb_bins=nb_bins)
            print(f"Loss: {loss.item()}")


def load_npz_images(file_path):
    with np.load(file_path) as data:
        images = data['images']
        return images
    
def load_random_data(H, W, n_images):
    return np.random.rand(H * W * n_images, 9)

def _ngp(device):
    H, W = 100,100  # 25 * 40 = 1000
    n_rand_images = 5
    ttr = 0.8
    batch_size = 64
    L = 16
    F = 2
    T = 2**19
    N_min = 16
    N_max = 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
    model = tinyNGP(T, Nl, 4, device, 3)
    #training_dataset = Tensor(torch.from_numpy(np.load('training_data_800x800.pkl',allow_pickle=True)).numpy())
    # # random data for quick testing
    #training_dataset = Tensor(load_random_data(H, W, n_images=n_images), device=device)
    # TinyNerf data, not to be confused with tinynerf
    images = load_npz_images('data/tiny_nerf_data.npz')
    train_images = images[:int(ttr * len(images))]
    print(f"Training images shape: {train_images.shape}")
    test_images = images[int(ttr * len(images)):]
    training_dataset = Tensor(torch.from_numpy(train_images).numpy(), requires_grad=False)
    # Get the lists of tensors.
    lookup_tables_list = list(model.lookup_tables.values())

    # Extract parameters from density_MLP
    density_MLP_params = []
    for layer in model.density_MLP:
        if isinstance(layer, nn.Linear):
            density_MLP_params.extend(nn.state.get_parameters(layer))

    # Extract parameters from color_MLP
    color_MLP_params = []
    for layer in model.color_MLP:
        if isinstance(layer, nn.Linear):
            color_MLP_params.extend(nn.state.get_parameters(layer))

    # Combine the lists of tensors
    params = lookup_tables_list + density_MLP_params + color_MLP_params
    
    # After creating the params list
    for param in params:
        if isinstance(param, Tensor):
            param.requires_grad = True


    # Now create the optimizer
    model_optimizer = nn.optim.Adam(params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8)

    data_loader = TinyDataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    train(model, model_optimizer, data_loader, nb_epochs=1, device=device, hn=2, hf=6, nb_bins=100, H=H, W=W)
    # unload the training data
    del training_dataset
    #testing_dataset = Tensor(torch.from_numpy(np.load('testing_data_800x800.pkl',allow_pickle=True)).numpy(), requires_grad=False)# this is the worst way to do this 
    #testing_dataset = Tensor(load_random_data(H, W, n_images=n_images), device=device)
    testing_dataset = Tensor(torch.from_numpy(test_images).numpy(), requires_grad=False)
    for img_index in range(len(testing_dataset) // (H * W)):
        test(2, 6, testing_dataset, img_index, nb_bins=192, H=H, W=W)
    # unload the testing data
    del testing_dataset
    
# One iteration of TinyNeRF (forward pass).
def iter_tinynerf(model,chunksize, height, width, focal_length, tform_cam2world,
                            near_thresh, far_thresh, depth_samples_per_ray,
                            encoding_function, get_minibatches_function):

    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length,
                                                tform_cam2world)
    
    # Sample query points along each ray
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = Tensor.cat(*predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = Tensor.reshape(radiance_field_flattened, unflattened_shape)

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)
    # After obtaining rgb_predicted in iter_tinynerf
    rgb_predicted = rgb_predicted.reshape(height, width, 3)
    return rgb_predicted
    
def _tinynerf(device):
    # Download sample data used in the official tiny_nerf example
    dpath = 'data/tiny_nerf_data.npz'
    if not os.path.exists(dpath):
        #!wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
        print("Downloading tiny_nerf_data.npz...")
        url = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
        r = requests.get(url)
        with open(dpath, 'wb') as f:
            f.write(r.content)

    # Load input images, poses, and intrinsics
    data = np.load(dpath)
    # torch.from_numpy(...).to(device) should be Tensor(...).to(device)
    # Images
    images = data["images"]
    # Camera extrinsics (poses)
    tform_cam2world = data["poses"]
    tform_cam2world = Tensor(tform_cam2world).to(device)
    # Focal length (intrinsics)
    focal_length = data["focal"]
    focal_length = Tensor(focal_length).to(device)

    # Height and width of each image
    height, width = images.shape[1:3]

    # Near and far clipping thresholds for depth values.
    near_thresh = 2.
    far_thresh = 6.

    # Hold one image out (for test).
    testimg, testpose = images[101], tform_cam2world[101]
    testimg = Tensor(testimg).to(device)

    # Map images to device
    images = Tensor(images).to(device)
    
    bool_display = False
    if bool_display:
        plt.imshow(testimg.detach().to('cpu').numpy())  
        plt.show()
    
    """
    Parameters for TinyNeRF training
    """

    # Number of functions used in the positional encoding (Be sure to update the 
    # model if this number changes).
    num_encoding_functions = 6
    # Specify encoding function.
    encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)
    # Number of depth samples along each ray.
    depth_samples_per_ray = 32

    # Chunksize (Note: this isn't batchsize in the conventional sense. This only
    # specifies the number of rays to be queried in one go. Backprop still happens
    # only after all rays from the current "bundle" are queried and rendered).
    chunksize = 16384  # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory.

    # Optimizer parameters
    lr = 5e-4
    num_iters = 1000

    # Misc parameters
    display_every = 50  # Number of iters after which stats are displayed

    """
    Model
    """
    model = TinyNerf(num_encoding_functions=num_encoding_functions)

    """
    Optimizer
    """
    optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)

    """
    Train-Eval-Repeat!
    """

    # Seed RNG, for repeatability
    seed = 9458
    Tensor.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []
    
    @TinyJit
    @Tensor.train()
    def step():
        # Randomly pick an image as the target.
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        rgb_predicted = iter_tinynerf(model, chunksize, height, width, focal_length,
                                                target_tform_cam2world, near_thresh,
                                                far_thresh, depth_samples_per_ray,
                                                encode, get_minibatches)

        # Compute mean-squared error between the predicted and target images. Backprop!
        #print(rgb_predicted.shape, target_img.shape)
        loss = ((rgb_predicted - target_img) ** 2).mean().backward()
        optimizer.step()
        return loss, target_img
    
    @Tensor.test()
    def get_test(target_img):
        # catching silent breaks
        # Render the held-out view
        rgb_predicted = iter_tinynerf(model, chunksize, height, width, focal_length,
                                    testpose, near_thresh, far_thresh,
                                    depth_samples_per_ray, encode, get_minibatches)
        # Compute PSNR
        loss = ((rgb_predicted - testimg) ** 2).mean()
        psnr = -10. * np.log10(loss.item())
        
        psnrs.append(psnr.item())
        iternums.append(i)
        # save image
        img = (rgb_predicted.detach().clip(0, 1)*255.).cast(dtypes.uint8).to('cpu').numpy()
        img = Image.fromarray(img)
        return loss, psnr, img
        
    for i in range(num_iters):
        # Run one iteration of TinyNeRF and backpropagate.
        loss, target_img = step()
        optimizer.zero_grad()
        # Display images/plots/stats
        if i % display_every == 0:
            loss_t, psnr_t, img_t = get_test(target_img)
            print(f"Iteration: {i}, PSNR: {psnr_t.item()}, Loss: {loss_t.item()}")
            img_t.save(f'images_from_training/img_{i}.png')
            
            

    print('Done!')
        
if __name__ == "__main__":
    device = 'cuda:0'
    _tinynerf(device)
    #_ngp(device)