from einops import rearrange
import torch


def Minkowski_distance(src, dst, p):
    """
    Calculate Minkowski distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point Minkowski distance, [B, N, M]
    """
    return torch.cdist(src, dst, p=p)


def knn_idx(xyz, new_xyz, K=3, p=2):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        (group_dist,group_idx): (grouped points distance, [B, S, K], grouped points index, [B, S, K])
    """

    sqrdists = Minkowski_distance(src=new_xyz, dst=xyz, p=p)
    group_dist, group_idx = torch.topk(
        sqrdists, K, dim=-1, largest=False, sorted=True)
    return group_dist, group_idx


def gather_idx(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S], --->[B,N,K]
    Return:
        new_points:, indexed points data, [B, S, C]---> [B, N, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def IGI2(x,idx):
    
    org_x = x

    neighbors = gather_idx(x,idx)
    neighbor_1st = neighbors[:,:,1,:] # B,N,C
    neighbor_2nd = neighbors[:,:,2,:] # B,N,C

    edge1 = neighbor_1st-org_x # B N C
    edge2 = neighbor_2nd-org_x # B N C

    edge1 = edge1/edge1.norm(dim=-1,keepdim=True)
    edge2 = edge2/edge2.norm(dim=-1,keepdim=True)


    
    gstd = neighbors.std(dim=2)
    gstd = gstd/gstd.norm(dim=-1,keepdim=True)


    normals = torch.cross(edge1, edge2, dim=-1) # B,N,3,f
    normals = normals/normals.norm(dim=-1,keepdim=True)

    new_pts = torch.cat((org_x, edge1, edge2, gstd, normals), -1)

    return new_pts



def laplace_point(x, K):
    _, idx = knn_idx(x, x, K, 2)
    return gather_idx(x, idx).mean(dim=2)-x


def laplace_vector(x, K):
    _, idx = knn_idx(x, x, K, 2)
    return torch.cat([x, gather_idx(x, idx).mean(dim=2)-x], dim=-1)


def GPC2(x, K=20, LV=True):
    x1 = x
    _, idx = knn_idx(x1, x1, K)
    x_geo = IGI2(x, idx)
    # Laplace
    if LV:
        return laplace_vector(x_geo, K)
    return x_geo


