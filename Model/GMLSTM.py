# GMLSTM class
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalisation factor for gaussian.
def gumbel_sample(x, axis=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z=torch.distributions.gumbel.Gumbel(0,1).sample(x.shape).to(device)

    #z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return torch.argmax(torch.log(x) + z,dim=axis)
def gaussian_distribution(y, mu, sigma):
  # braodcast subtraction with mean and normalization to sigma
  result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
  result = - 0.5 * (result * result)
  return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI

def gmlstm_loss_function(out_pi, out_sigma, out_mu, y):
  epsilon = 1e-7
  result = gaussian_distribution(y, out_mu, out_sigma) * out_pi
  result = torch.sum(result, dim=1)
  result = - torch.log(epsilon + result)
  return torch.mean(result)

def gmlstm_loss_function_rnn(out_pi, out_sigma, out_mu, y):
  epsilon = 1e-7
  result = gaussian_distribution(y, out_mu, out_sigma) * out_pi
  result = torch.sum(result, dim=2)
  result = - torch.log(epsilon + result)
  return torch.mean(result)

def elu_plus_one_plus_epsilon(x):
  """ELU activation with a very small addition to help prevent
  NaN in loss."""
  return nn.functional.elu(x) + 1 + 1e-8

class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        #sigma = torch.exp(self.z_sigma(z_h))
        sigma = elu_plus_one_plus_epsilon(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu

class GMLSTM(nn.Module):
    def __init__(self, mdn_hidden, n_gaussians,in_feature=1):
        super(GMLSTM, self).__init__()
        self.rnn = nn.LSTM(in_feature, mdn_hidden, 3, batch_first=True,bidirectional = True,dropout=0.2)
        #print(mdn_hidden,n_gaussians)
        # self.z_h = nn.Sequential(
        #     nn.LSTM(1, mdn_hidden, 1, batch_first=True),
        #     nn.Tanh()
        # )
        self.z_pi = nn.Linear(2*mdn_hidden, n_gaussians)
        self.z_sigma = nn.Linear(2*mdn_hidden, n_gaussians)
        self.z_mu = nn.Linear(2*mdn_hidden, n_gaussians)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        z_h = nn.Tanh()(rnn_out)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        #pi=nn.functional.gumbel_softmax(self.z_pi(z_h),tau=5,hard=True,dim=-1)
        #sigma = torch.exp(self.z_sigma(z_h))
        sigma = elu_plus_one_plus_epsilon(self.z_sigma(z_h))
        mu = self.z_mu(z_h)

        return pi, sigma, mu
class GMGRU(nn.Module):
    def __init__(self, mdn_hidden, n_gaussians,in_feature=1):
        super(GMGRU, self).__init__()
        self.rnn = nn.GRU(in_feature, mdn_hidden, 3, batch_first=True,bidirectional = True,dropout=0.2)
        #print(mdn_hidden,n_gaussians)
        # self.z_h = nn.Sequential(
        #     nn.LSTM(1, mdn_hidden, 1, batch_first=True),
        #     nn.Tanh()
        # )
        self.z_pi = nn.Linear(2*mdn_hidden, n_gaussians)
        self.z_sigma = nn.Linear(2*mdn_hidden, n_gaussians)
        self.z_mu = nn.Linear(2*mdn_hidden, n_gaussians)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        z_h = nn.Tanh()(rnn_out)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        #pi=nn.functional.gumbel_softmax(self.z_pi(z_h),tau=5,hard=True,dim=-1)
        #sigma = torch.exp(self.z_sigma(z_h))
        sigma = elu_plus_one_plus_epsilon(self.z_sigma(z_h))
        mu = self.z_mu(z_h)

        return pi, sigma, mu
NHIDDEN = 100 # hidden units
KMIX = 20 # number of mixtures

