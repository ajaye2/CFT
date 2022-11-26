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



def quadratic_contrastive_loss(z1, f1, delta=1, type_sim='square'):
    '''
        z1 is a batch containing the representations from the dilated conv net
        f1 is a batch containing the expert features 
    '''
    loss                    = torch.tensor(0., device=z1.device)
    B, T                    = z1.size(0), z1.size(1)
    max_diff_norm_of_feat   = get_max_norm(f1)
    dij_mu                  = get_euclidean_mean(z1)

    for i in range(B):  
        xi   = z1[i]
        fi   = f1[i]
        for j in range(B): 
            xj   = z1[j]
            fj   = f1[j]
            loss += get_quad_loss_helper(xi, xj, fi, fj, dij_mu[i], max_diff_norm_of_feat, delta, type_sim)
        
    loss /= B**2

    return loss



def expclr_loss(z1, f, temp=1, delta=1, type_sim='square'):
    '''
        z1 is a batch containing the representations from the dilated conv net
        f1 is a batch containing the expert features 
    '''

    # TODO: Make sure that we getting the loss right, 
    # TODO: Check why we getting NAN losses 

    loss                    = torch.tensor(0., device=z1.device)
    B, T                    = z1.size(0), z1.size(1)
    max_diff_norm_of_feat   = get_max_norm(f)
    dij_mu                  = get_euclidean_mean(z1)

    for i in range(B):  
        xi   = z1[i]
        fi   = f[i]
        for j in range( B): 
            xj  = z1[j]
            fj  = f[j]

            lij = get_quad_loss_helper(xi, xj, fi, fj, dij_mu[i], max_diff_norm_of_feat, delta, type_sim)
            lij = torch.exp(lij / temp)
            # lij /= B**2

            loss += lij
    loss /= B**2

    loss = temp * torch.log( loss )
    return loss
    
def get_quad_loss_helper(xi, xj, fi, fj, dij_mu, max_diff_norm_of_exp_feat, delta, type_sim):
        sij  = similarity_measure(fi, fj, max_diff_norm_of_exp_feat, type=type_sim)
        dij  = torch.norm(xi - xj) / dij_mu # TODO: Add explanation for dividing by mean 
        loss = ( (1-sij)*delta - dij ) ** 2
        return loss

def get_euclidean_mean(z1):
    B, T     = z1.size(0), z1.size(1)
    means = {}
    for i in range(B):  
        xi   = z1[i]
        means[i] = torch.tensor(0., device=z1.device)
        for j in range(B): 
            xj  = z1[j]
            dij = torch.norm(xi - xj)
            means[i]+=dij
        means[i] /= B
    return means

def similarity_measure(f1, f2, max_diff_norm, type='square', sigma=0.01):
    sij = 1 - ( torch.norm(f1 - f2)  / max_diff_norm)
    if type=='square':
        return sij ** 2
    elif type=='exp':
        # TODO: Double check if this is right
        return torch.exp( -(torch.norm(f1 - f2) ** 2) / sigma) 
    else:
        return sij


def get_max_norm(f1):
    '''
        f1 is a batch containing the expert features 
    '''
    max_diff =  torch.tensor(0., device=f1.device)
    B        =  f1.size(0)

    # TODO: Make get_max_norm more efficient
    for i in range(B):
        for j in range( B ):
            diff = f1[i] - f1[j]
            norm_diff = torch.norm(diff) 
            if max_diff < norm_diff:
                max_diff = norm_diff

    return max_diff