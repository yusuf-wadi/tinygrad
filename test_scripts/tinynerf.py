from tinygrad import Device
print(Device.DEFAULT)
import torch #TODO replace torch where you can
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import tinygrad.nn as nn
from tinygrad.tensor import Tensor, dtypes
from tinygrad.function import Relu, Sigmoid
from PIL import Image
from typing import List, Dict

def test(hn, hf, dataset, img_index, chunk_size=20, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    px_values = []   # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        px_values.append(render_rays(model, ray_origins_, ray_directions_,
                                     hn=hn, hf=hf, nb_bins=nb_bins))
    img = Tensor.cat(px_values).data.cpu().numpy().reshape(H, W, 3)
    img = (img.clip(0, 1)*255.).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f'novel_views/img_{img_index}.png')

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
            Relu(device=self.device),
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
        print(out)
        return Tensor.cat(*out, dim=1)

    def __call__(self, x: Tensor, d):
        
        x /= self.aabb_scale
        assert x.shape[1] == 3, "Input tensor x must have shape (N, 3)"
        mask = (x[:, 0].abs() < .5).int() & (x[:, 1].abs() < .5).int() & (x[:, 2].abs() < .5).int()
        mask = mask.to(x.device)  # Ensure mask is on the same device as x
        x += 0.5 # x in [0, 1]^3
        
        color = Tensor.zeros((x.shape[0], 3), device=x.device)
        log_sigma = (Tensor.zeros((x.shape[0]), device=x.device) - 1e5).contiguous() # when we compute sigma it will be 0 (?)
        features = Tensor.empty((x[mask].shape[0], self.F * len(self.Nl)), device=x.device)
             
        for i, N in enumerate(self.Nl):
            # compute vertices
            floor = Tensor.floor(x[mask] * N)
            #print(floor.numpy())
            ceil = Tensor.ceil(x[mask] * N)
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

                        
        xi = self.positional_encoding(d[mask])
        h = features.sequential(self.density_MLP)
        log_sigma[mask] = h[:, 0]
        color[mask] = Tensor.cat(xi, h[:, 1:], dim=1).sequential(self.color_MLP)
        return color, Tensor.exp(log_sigma)

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

def cumprod(input, dim, dtype=None):
    # Ensure input is a tinygrad Tensor
    if not isinstance(input, Tensor):
        input = Tensor(input)

    # Get the shape of the input tensor
    shape = input.shape

    # Reshape the tensor to bring the dimension to be cumulated over to the end
    permutation = list(range(len(shape)))
    permutation.remove(dim)
    permutation.append(dim)
    reshaped_input = input.transpose(permutation)

    # Compute the cumulative product along the last dimension
    reshaped_output = Tensor(reshaped_input.numpy().cumprod(axis=-1))

    # Reshape the output tensor back to the original shape
    output = reshaped_output.transpose(permutation)

    # Cast the output to the specified data type if provided
    if dtype is not None:
        output = output.cast(dtype)

    return output


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = cumprod(alphas, dim=1)
    return Tensor.cat((Tensor.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
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
    c = (weights * colors.reshape(x.shape)).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)

def train(nerf_model: tinyNGP, optimizer: nn.optim.LAMB, data_loader: TinyDataLoader, device='cpu', hn=0, hf=1, nb_epochs=10, nb_bins=192, H=400, W=400):
    for _ in range(nb_epochs):
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            gt_px_values = batch[:, 6:].to(device)

            pred_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((gt_px_values - pred_px_values) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
if __name__ == "__main__":
    device = 'cuda:0'
    #training_dataset = Tensor(torch.from_numpy(np.load('training_data_800x800.pkl',allow_pickle=True)).numpy())
    #testing_dataset = Tensor(torch.from_numpy(np.load('testing_data_800x800.pkl',allow_pickle=True)).numpy(), requires_grad=False)# this is the worst way to do this
    # dummy data for quick testing
    training_dataset = Tensor(np.random.rand(1000, 9), device=device)
    testing_dataset = Tensor(np.random.rand(1000, 9), device=device)
    L = 16
    F = 2
    T = 2**19
    N_min = 16
    N_max = 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
    model = tinyNGP(T, Nl, 4, device, 3)

    # Get the lists of tensors
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

    # Create the optimizer with the combined parameter list
    model_optimizer = nn.optim.Adam(params, lr=1e-3)

    data_loader = TinyDataLoader(training_dataset, batch_size=2**14, shuffle=True)
    train(model, model_optimizer, data_loader, nb_epochs=1, device=device, hn=2, hf=6, nb_bins=192, H=800, W=800)
    for img_index in range(200):
        test(2, 6, testing_dataset, img_index, nb_bins=192, H=800, W=800)