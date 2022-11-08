import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss



def quadratic_contrastive_loss(z1, f1, delta=0.01):
    '''
        z1 is a batch containing the representations from the dilated conv net
        f1 is a batch containing the expert features 
    '''
    loss     = torch.tensor(0., device=z1.device)
    B, T     = z1.size(0), z1.size(1)
    max_diff = get_max_norm(f1)
    dij_mu   = get_euclidean_mean(z1)

    for i in range(B):  
        xi   = z1[i]
        for j in range(i+1, B): 
            xj  = z1[j]
            sij = similarity_measure(xi, xj, max_diff)
            dij = torch.norm(xi - xj) / dij_mu
            loss += ( (1-sij)*delta - dij ) ** 2

    loss /= B**2

    return loss


def expclr_loss(z1, f1, temp=0.5, delta=0.01):
    '''
        z1 is a batch containing the representations from the dilated conv net
        f1 is a batch containing the expert features 
    '''
    loss     = torch.tensor(0., device=z1.device)
    B, T     = z1.size(0), z1.size(1)
    max_diff = get_max_norm(f1)
    dij_mu   = get_euclidean_mean(z1)

    for i in range(B):  
        xi   = z1[i]
        for j in range(i+1, B): 
            xj  = z1[j]
            sij = similarity_measure(xi, xj, max_diff)
            dij = torch.norm(xi - xj)/dij_mu

            lij = ( (1-sij)*delta - dij ) ** 2 
            lij /= temp
            lij = torch.exp(lij)
            lij /= B**2

            loss += lij
    loss = temp * torch.log( loss )
    return loss
    

def get_euclidean_mean(z1):
    mean     = torch.tensor(0., device=z1.device)
    B, T     = z1.size(0), z1.size(1)

    for i in range(B):  
        xi   = z1[i]
        for j in range(i+1, B): 
            xj  = z1[j]
            dij = torch.norm(xi - xj)
            mean += dij
    mean /= B
    return mean

def similarity_measure(z1, z2, max_diff, type='square', sigma=0.01):
    sij = 1 - ( torch.norm(z1 - z2)  / max_diff)
    if type=='square':
        return sij ** 2
    elif type=='exp':
        # TODO
        pass 
    else:
        return sij


def get_max_norm(f1):
    '''
        f1 is a batch containing the expert features 
    '''
    max_diff =  torch.tensor(0., device=f1.device)

    # TODO: Max more efficient

    for i in range( len(f1) ):
        for j in range( i+1, len(f1) ):
            if i == j: continue
            diff = f1[i] - f1[j]
            norm_diff = torch.norm(diff) 
            if max_diff < norm_diff:
                max_diff = norm_diff

    return max_diff