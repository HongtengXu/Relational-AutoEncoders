import torch
from torch import nn
from torch import optim
import Methods.evaluation as evaluation
import Methods.models as method


# regularizer used in proximal-relational autoencoder
class Prior(nn.Module):
    def __init__(self, data_size: list):
        super(Prior, self).__init__()
        # data_size = [num_component, z_dim]
        self.data_size = data_size
        self.number_components = data_size[0]
        self.output_size = data_size[1]
        self.linear1 = nn.Linear(self.number_components, self.output_size, bias=True)
        self.linear2 = nn.Linear(self.number_components, self.output_size, bias=False)
        self.idle_input = torch.eye(self.number_components, self.number_components, requires_grad=False)

    def forward(self):
        mu = self.linear1(self.idle_input)
        logvar = self.linear2(self.idle_input)
        return mu, logvar


def sum_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor):
    """
    Returns the matrix of "x_i + y_j".
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :return: [R, C, D] sum matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    return x_col + y_row


def prod_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor):
    """
    Returns the matrix of "x_i * y_j".
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :return: [R, C, D] sum matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    return x_col * y_row


def distance_tensor(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C, D] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.abs(x_col - y_row) ** p
    return distance


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def distance_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, logvar_src: torch.Tensor, logvar_dst: torch.Tensor):
    """
    Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances
    :param mu_src: [R, D] matrix, the means of R Gaussian distributions
    :param mu_dst: [C, D] matrix, the means of C Gaussian distributions
    :param logvar_src: [R, D] matrix, the log(variance) of R Gaussian distributions
    :param logvar_dst: [C, D] matrix, the log(variance) of C Gaussian distributions
    :return: [R, C] distance matrix
    """
    std_src = torch.exp(logvar_src)
    std_dst = torch.exp(logvar_dst)
    distance_mean = distance_matrix(mu_src, mu_dst, p=2)
    distance_var = torch.sum(sum_matrix(std_src, std_dst) - 2 * (prod_matrix(std_src, std_dst) ** 0.5), 2)
    return distance_mean + distance_var


def tensor_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, logvar_src: torch.Tensor, logvar_dst: torch.Tensor):
    """
    Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances
    :param mu_src: [R, D] matrix, the means of R Gaussian distributions
    :param mu_dst: [C, D] matrix, the means of C Gaussian distributions
    :param logvar_src: [R, D] matrix, the log(variance) of R Gaussian distributions
    :param logvar_dst: [C, D] matrix, the log(variance) of C Gaussian distributions
    :return: [R, C, D] distance tensor
    """
    std_src = torch.exp(logvar_src)
    std_dst = torch.exp(logvar_dst)
    distance_mean = distance_tensor(mu_src, mu_dst, p=2)
    distance_var = sum_matrix(std_src, std_dst) - 2 * (prod_matrix(std_src, std_dst) ** 0.5)
    return distance_mean + distance_var


def cost_mat(cost_s: torch.Tensor,
             cost_t: torch.Tensor,
             tran: torch.Tensor) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)

    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:

    f1(a) = a^2,
    f2(b) = b^2,
    h1(a) = a,
    h2(b) = 2b

    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have

    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    f1_st = torch.sum(cost_s ** 2, dim=1, keepdim=True) / cost_s.size(0)
    f2_st = torch.sum(cost_t ** 2, dim=1, keepdim=True) / cost_t.size(0)
    tmp = torch.sum(sum_matrix(f1_st, f2_st), dim=2)
    cost = tmp - 2 * cost_s @ tran @ torch.t(cost_t)
    return cost


def gwd_estimation(cost_s: torch.Tensor,
                   cost_t: torch.Tensor,
                   num_layer: int = 20):
    ns = cost_s.size(0)
    nt = cost_t.size(0)
    p_s = torch.ones(ns, 1) / ns
    p_t = torch.ones(nt, 1) / nt
    tran = torch.ones(ns, nt) / (ns * nt)
    dual = torch.ones(ns, 1) / ns
    for m in range(num_layer):
        cost = cost_mat(cost_s, cost_t, tran)
        # cost /= torch.max(cost)
        kernel = torch.exp(-10 * cost / torch.max(cost)) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(5):
            dual = p_s / (kernel @ b)
            b = p_t / (torch.t(kernel) @ dual)
        tran = (dual @ torch.t(b)) * kernel

    d_gw = (cost_mat(cost_s, cost_t, tran) * tran).sum()
    return d_gw, tran


def wd_estimation(cost: torch.Tensor, num_layer: int = 20):
    ns = cost.size(0)
    nt = cost.size(1)
    p_s = torch.ones(ns, 1) / ns
    p_t = torch.ones(nt, 1) / nt
    tran = torch.ones(ns, nt) / (ns * nt)

    dual = torch.ones(ns, 1) / ns
    gkernel = torch.exp(-10 * cost / torch.max(cost))
    for m in range(num_layer):
        kernel = gkernel * tran
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(5):
            dual = p_s / (kernel @ b)
            b = p_t / (torch.t(kernel) @ dual)
        tran = (dual @ torch.t(b)) * kernel

    d_w = (cost * tran).sum()
    return d_w, tran


def W_discrepancy(mu1, mu2, logvar1, logvar2):
    cost = distance_gmm(mu1, mu2, logvar1, logvar2)
    _, tran = wd_estimation(cost)
    return (cost * tran.detach().data).sum()


def GW_discrepancy(mu1, logvar1):
    cost1 = distance_gmm(mu1, mu1, logvar1, logvar1)
    cost2 = torch.ones(size=cost1.size(), requires_grad=False) - torch.eye(cost1.size(0), requires_grad=False)
    _, tran = gwd_estimation(cost1, cost2)
    return (cost_mat(cost1, cost2, tran.detach().data) * tran.detach().data).sum()


def train(model, prior, train_loader, optimizer_model, optimizer_prior, device, epoch, args):
    model.train()
    prior.eval()
    train_rec_loss = 0
    train_reg_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer_model.zero_grad()
        recon_batch, z, mu, logvar = model(data)
        z_mu, z_logvar = prior()
        rec_loss = method.loss_function(recon_batch, data, args.rec_type)
        reg_loss = args.gamma * W_discrepancy(mu, z_mu, logvar, z_logvar)
        loss = rec_loss + reg_loss
        loss.backward()
        train_rec_loss += rec_loss.item()
        train_reg_loss += reg_loss.item()
        optimizer_model.step()
        if batch_idx % args.log_interval == 0:
            print('Train Model Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))

    model.eval()
    prior.train()
    train_w_loss = 0
    train_gw_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer_prior.zero_grad()
        recon_batch, z, mu, logvar = model(data)
        z_mu, z_logvar = prior()
        w_loss = args.gamma * W_discrepancy(mu, z_mu, logvar, z_logvar)
        gw_loss = args.beta * GW_discrepancy(z_mu, z_logvar)
        loss = w_loss + gw_loss
        loss.backward()
        train_w_loss += w_loss.item()
        train_gw_loss += gw_loss.item()
        optimizer_prior.step()
        if batch_idx % args.log_interval == 0:
            print('Train Prior Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))

    print('====> Epoch: {} Average RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        epoch, train_w_loss / len(train_loader.dataset), train_gw_loss / len(train_loader.dataset),
        (train_w_loss + train_gw_loss) / len(train_loader.dataset)))


def test(model, prior, test_loader, device, args):
    model.eval()
    prior.eval()
    test_rec_loss = 0
    test_reg_loss = 0
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, z, mu, logvar = model(data)
            z_mu, z_logvar = prior()
            rec_loss = method.loss_function(recon_batch, data, args.rec_type)
            reg_loss = args.gamma * W_discrepancy(mu, z_mu, logvar, z_logvar) + \
                args.beta * GW_discrepancy(z_mu, z_logvar)
            test_rec_loss += rec_loss.item()
            test_reg_loss += reg_loss.item()
            test_loss += (rec_loss.item() + reg_loss.item())

    test_rec_loss /= len(test_loader.dataset)
    test_reg_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        test_rec_loss, test_reg_loss, test_loss))
    return test_rec_loss, test_reg_loss, test_loss


def train_model(model, prior, train_loader, test_loader, device, args):
    loss_list = []
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_prior = optim.Adam(prior.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        train(model, prior, train_loader, optimizer_model, optimizer_prior, device, epoch, args)
        test_rec_loss, test_reg_loss, test_loss = test(model, prior, test_loader, device, args)
        loss_list.append([test_rec_loss, test_reg_loss, test_loss])
        if epoch % 5 == 0:
            evaluation.interpolation_2d(model, test_loader, device, epoch, args)
            prior.eval()
            z_p_mean, z_p_logvar = prior()
            evaluation.sampling(model, device, epoch, args, prior=[z_p_mean, z_p_logvar])
            evaluation.reconstruction(model, test_loader, device, epoch, args)
    return loss_list
