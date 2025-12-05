import torch
import torch.nn.functional as F

def cdf_quantile_match_pytorch(s, t):
    """
    s: source distribution (B, 1, B, H, W)
    t: target distribution (B, 1, B, H, W)
    """

    B, C, H, W, D = s.shape


    s = s.view(B, -1)
    t = t.view(B, -1)

    N = s.shape[-1]

    sorted_indices_s = torch.argsort(s, dim=-1) 
    
    s_ranks = torch.empty_like(sorted_indices_s, dtype=torch.float32).to(s.device)
    s_ranks[:, sorted_indices_s] = torch.arange(N, dtype=torch.float32).to(s.device) + 1
    
    t_sorted, _ = torch.sort(t, dim=-1)
    t_indices = (s_ranks - 1).long()

    s_new = t_sorted[:, t_indices]

    return s_new


def compute_3d_sobel_gradient(input_tensor):
    """
    compute 3d volume Sobel gradient
    """
    sobel_x = torch.tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ], dtype=torch.float32).view(1, 1, 3, 3, 3)
    
    sobel_y = torch.tensor([
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ], dtype=torch.float32).view(1, 1, 3, 3, 3)
    
    sobel_z = torch.tensor([
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    ], dtype=torch.float32).view(1, 1, 3, 3, 3)
    
    device = input_tensor.device
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)
    sobel_z = sobel_z.to(device)
    
    padded_input = F.pad(input_tensor, (1, 1, 1, 1, 1, 1), mode='replicate')
    
    grad_x = F.conv3d(padded_input, sobel_x, padding=0)
    grad_y = F.conv3d(padded_input, sobel_y, padding=0)
    grad_z = F.conv3d(padded_input, sobel_z, padding=0)
    
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    return gradient_magnitude


def extract_neighborhood_vectors(input_tensor, kernel_size=11):
    """
    kernel_size^3 neighborhood vector
    """
    B, C, H, W, D = input_tensor.shape
    
    pad_size = kernel_size // 2
    
    padded_input = F.pad(input_tensor, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='replicate')
    
    output = torch.zeros(B, kernel_size**3, H, W, D, device=input_tensor.device)
    
    idx = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                output[:, idx, :, :, :] = padded_input[:, :, i:i+H, j:j+W, k:k+D]
                idx += 1
                
    return output


def sobel_gradient_to_neighborhood_vector(input_tensor, kernel_size=11):

    gradient = compute_3d_sobel_gradient(input_tensor)
    
    neighborhood_vectors = extract_neighborhood_vectors(gradient, kernel_size)
    
    return neighborhood_vectors