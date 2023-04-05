import torch
from torch.autograd import Variable
from torch.nn import functional as F


from geoopt.manifolds.stereographic import PoincareBall


def euclidean_dist(x, y):
    """
    Computes euclidean distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def Poincare_dist(x, y, c=1.0):
    """
    Computes Poincare distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    manifold = PoincareBall(c=c)

    # x = manifold.projx(x)
    # y = manifold.projx(y)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    hx = manifold.projx(x)
    hy = manifold.projx(y)

    return manifold.dist2(hx, hy)


class ProtoNet(torch.nn.Module):
    def __init__(self, encoder, c=None, manifold="poincare"):
        super(ProtoNet, self).__init__()

        """
    Args:
        encoder : CNN encoding the images in sample
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
    """
        self.encoder = encoder.cuda()
        self.c = c
        self.manifold = manifold

    def set_forward_loss(self, sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim)) 
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat
        """
        sample_images = sample['pointcloud'].cuda()
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        x_support = sample_images[:, :n_support]
        x_query = sample_images[:, n_support:]

        # target indices are 0 ... n_way-1
        target_inds = torch.arange(0, n_way).view(
            n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda()

        # encode images of the support and the query set
        x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                       x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)  # usually 64
        z_query = z[n_way*n_support:]
        z_support = z[:n_way*n_support].view(n_way, n_support, z_dim)
        z_proto = z_support.mean(1)

        # compute distances
        if self.manifold.lower()  == 'euclidean':
            dists = euclidean_dist(z_query, z_proto)


        else:
            dists = Poincare_dist(z_query, z_proto, c=self.c)

        # compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat,
            "z_query": z_query,
            "z_proto": z_proto,
            "z_support": z_support.view(n_way*n_support, z_dim),
            "labels": target_inds 
        }