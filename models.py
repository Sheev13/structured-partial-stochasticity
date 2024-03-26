import torch
from torch import nn
from typing import List, Optional, Tuple
from weight_masks import get_mask, apply_mask

# TODO: split into BNN and NN classes


class MeanFieldLayer(nn.Module):
    """Represents a layer of a BNN under mean-field variational inference

    Args:
        input_dim: the number of input features
        output_dim: the number of output features
        prior_std: the standard deviation of the Gaussian weight prior
        activation: the nonlinear activation function that acts elementwise on the layer output
                and the initial standard deviation of the variational posterior
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: nn.Module,
        prior_std: float = 1.0,
        scale_prior: bool = False,
        asymmetric_weights: bool = False,
        odd: bool = False,
        minimal_mask: bool = True,
        c: Optional[float] = None,
        map_weights: Optional[int] = None,
        random_mask: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.scale_prior = scale_prior
        self.asymmetric_weights = asymmetric_weights
        self.odd = odd
        self.minimal_mask = minimal_mask
        self.random_mask = random_mask
        if c is not None:
            self.c = c.detach()
        else:
            self.c = None

        if scale_prior:
            prior_std = prior_std / torch.sqrt(torch.tensor(input_dim))

        if map_weights is not None:
            self.w = map_weights
        else:
            self.w = nn.Parameter(torch.randn((input_dim + 1, output_dim)) * 1e-1)
        self.w_mu = nn.Parameter(torch.randn_like(self.w) * 1e-1)
        self.w_log_std = nn.Parameter(torch.randn_like(self.w) * 0.1 - 3)

        self.p = torch.distributions.Normal(
            torch.zeros_like(self.w),
            torch.ones_like(self.w) * prior_std,
        )

        if self.asymmetric_weights:
            self.init_weights_mask, self.kl_mask = get_mask(
                self.input_dim + 1,
                self.output_dim,
                self.odd,
                self.c,
                self.minimal_mask,
                map_weights,
                self.random_mask,
            )

    @property
    def w_std(self):
        return self.w_log_std.exp()

    @property
    def q(self):
        return torch.distributions.Normal(self.w_mu, self.w_std)

    def weights_mask(self, c: Optional[nn.Parameter] = None):
        if c is None or self.init_weights_mask.abs().max().abs().max() == 0:
            return self.init_weights_mask
        else:
            return (self.init_weights_mask / self.init_weights_mask.abs().max()) * c

    def kl(self):
        if self.asymmetric_weights:
            return (
                torch.distributions.kl.kl_divergence(self.q, self.p) * self.kl_mask
            ).sum()
        else:
            return torch.distributions.kl.kl_divergence(self.q, self.p).sum()

    def forward(
        self,
        x: torch.Tensor,
        variational: bool = False,
        c: Optional[nn.Parameter] = None,
    ):
        assert (
            len(x.shape) == 3
        ), "x should be shape (num_samples, batch_size, input_dim)."
        assert x.shape[-1] == self.input_dim
        num_samples = x.shape[0]

        # augment each x vector with a 1 to be used as bias coefficient
        x_aug = torch.cat((x, torch.ones(x.shape[:-1] + (1,))), dim=-1)

        if variational:
            num_samples = x.shape[0]
            w = self.q.rsample((num_samples,))
        else:
            w = self.w.unsqueeze(0).repeat(num_samples, 1, 1)

        if self.asymmetric_weights and variational:
            w = apply_mask(w, self.weights_mask(c), self.c == 0)

        return self.activation(x_aug @ w)  # shape (num_samples, batch_size, output_dim)


class MeanFieldBNN(nn.Module):
    """Represents a BNN under mean-field variational inference

    Args:
        dims: a list of the number of units in each layer of the network
        prior_std: the standard deviation of the Gaussian weight priors
        activation: the nonlinear activation function that acts elementwise between layers
        likelihood_std: the standard deviation of the Gaussian likelihood
                and the initial standard deviation of the variational posterior
    """

    def __init__(
        self,
        dims: List[int],
        prior_std: float = 1.0,
        activation: nn.Module = nn.Tanh(),
        likelihood_std: float = 0.05,
        scale_prior: bool = False,
        asymmetric_weights: bool = False,
        minimal_mask: bool = True,
        c: Optional[float] = None,
        map_weights: Optional[List[torch.Tensor]] = None,
        train_c: bool = False,
        random_mask: bool = False,
    ):
        super().__init__()

        self.likelihood_std = likelihood_std
        self.layers = nn.ModuleList()
        self.prior_std = prior_std
        self.activation = activation
        self.scale_prior = scale_prior
        self.asymmetric_weights = asymmetric_weights
        self.minimal_mask = minimal_mask
        self.random_mask = random_mask
        if c is not None:
            self.c = nn.Parameter(torch.tensor(c), requires_grad=train_c)
        else:
            self.c = None
        self.train_c = train_c

        if map_weights is None:
            map_weights = [None] * (len(dims) - 1)

        for i in range(len(dims) - 2):
            self.layers.append(
                MeanFieldLayer(
                    dims[i],
                    dims[i + 1],
                    self.activation,
                    self.prior_std,
                    self.scale_prior,
                    self.asymmetric_weights,
                    bool(i % 2),
                    self.minimal_mask,
                    self.c,
                    map_weights[i],
                    self.random_mask,
                )
            )
        self.layers.append(
            MeanFieldLayer(
                dims[-2],
                dims[-1],
                nn.Identity(),
                self.prior_std,
                self.scale_prior,
                asymmetric_weights,
                bool((len(dims) - 2) % 2),
                self.minimal_mask,
                self.c,
                map_weights[len(dims) - 2],
                self.random_mask,
            )
        )

        if asymmetric_weights:
            for i in range(len(dims)):
                if i == 0:
                    print(f"FC neurons in layer 1: {dims[0]}")
                elif i == len(dims) - 1:
                    print(f"FC neurons in layer {len(dims)}: {dims[-1]}")
                else:
                    n = dims[i] - dims[i - 1] - dims[i + 1] + 2
                    print(f"FC neurons in layer {i + 1}: {max(0, n)}")
        else:
            for i in range(len(dims)):
                print(f"FC neurons in layer {i + 1}: {dims[i]}")

    def forward(
        self,
        x: torch.Tensor,
        variational: bool = False,
        num_samples: int = 1,
    ):
        while len(x.shape) < 3:
            x = x.unsqueeze(0)

        x = x.repeat(num_samples, 1, 1)
        # x.shape is now (num_samples, batch_dim, input_dim)

        if not variational:
            for layer in self.layers:
                x = layer(x)

        else:
            for layer in self.layers:
                if self.train_c:
                    x = layer(x, variational=True, c=self.c)
                else:
                    x = layer(x, variational=True)

        return x

    def map_loss(  # maximum a posteriori loss for deterministic training
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        while len(y.shape) < 3:
            y = y.unsqueeze(0)

        preds = self(x)

        gaussian_likelihood = torch.distributions.Normal(preds, self.likelihood_std)
        log_likelihood = gaussian_likelihood.log_prob(y).sum()

        log_prior = sum([layer.p.log_prob(layer.w).sum() for layer in self.layers])

        log_posterior = log_prior + log_likelihood

        metrics = {
            "log posterior": log_posterior.detach(),
            "log likelihood": log_likelihood.detach(),
            "log prior": log_prior.detach(),
        }

        return -log_posterior, metrics

    def elbo_loss(  # variational objective for approx inference
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_samples: int = 1,
    ):
        while len(y.shape) < 3:
            y = y.unsqueeze(0)

        preds = self(x, True, num_samples)

        gaussian_likelihood = torch.distributions.Normal(preds, self.likelihood_std)
        exp_ll = gaussian_likelihood.log_prob(y).mean(0).sum()

        kl = torch.tensor(0.0)
        for layer in self.layers:
            kl += layer.kl()

        elbo = exp_ll - kl

        metrics = {
            "elbo": elbo.detach(),
            "exp ll": exp_ll.detach(),
            "kl": kl.detach(),
        }

        return -elbo, metrics

    def evaluate(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        variational: bool = False,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            test_preds = self(
                x_test,
                variational,
                num_samples,
            )
            mean_pred_rmse = ((test_preds.mean(0) - y_test.squeeze(0)) ** 2).mean(0).sqrt().mean().detach()
            mlpp = self._mean_log_posterior_predictive(test_preds, y_test).detach()

        return mean_pred_rmse, mlpp

    def _mean_log_posterior_predictive(
        self, preds: torch.Tensor, y_test: torch.Tensor
    ) -> torch.Tensor:
        """Mean log posterior predictive probability, mean taken over batch and dimensions (in log space)."""
        num_samples = preds.shape[0]
        gaussian_likelihood = torch.distributions.Normal(preds, self.likelihood_std)
        ll = gaussian_likelihood.log_prob(
            y_test
        )  # shape (num_samples, batch_size, input_dim)
        return (
            torch.logsumexp(ll, dim=0) - torch.log(torch.tensor(num_samples))
        ).mean()
