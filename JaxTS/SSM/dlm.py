from .base import StateSpaceModel
import jax.numpy as jnp
import jax
from functools import reduce

class DLM(StateSpaceModel):
    def __init__(self, y, X, trend_order=None, seasonal_periods=None, prior_state_mean=None, prior_state_cov=None, observation_cov=None, state_cov=None):
	    self.trend_order = trend_order
        self.seasonal_periods = seasonal_periods
        self.y = jnp.array(y)
        self.n_series, self.n_timesteps = self.y.shape
        self.X = jnp.array(X) if X is not None else None

        # Initialize the state transition matrix F
        self.F = self.build_state_transition_matrix()

        # Initialize the observation matrix
        self.observation_matrix = self.build_observation_matrix()

    def build_state_transition_matrix(self):
        components = []

        # Trend component
        if self.trend_order is not None:
            trend_matrix = jnp.eye(self.trend_order, k=-1)
            trend_matrix = jax.ops.index_update(trend_matrix, jax.ops.index[0, :], 1)
            components.append(trend_matrix)

        # Seasonal component with Fourier series
        if self.seasonal_periods is not None:
            seasonal_matrices = []
            for s in self.seasonal_periods:
                m = len(s) * 2  # Number of Fourier terms (sine and cosine)
                S = jnp.zeros((m, m))
                for i in range(len(s)):
                    S = jax.ops.index_update(S, jax.ops.index[2 * i, 2 * i], jnp.cos(2 * jnp.pi / s[i]))
                    S = jax.ops.index_update(S, jax.ops.index[2 * i, 2 * i + 1], jnp.sin(2 * jnp.pi / s[i]))
                    S = jax.ops.index_update(S, jax.ops.index[2 * i + 1, 2 * i], -jnp.sin(2 * jnp.pi / s[i]))
                    S = jax.ops.index_update(S, jax.ops.index[2 * i + 1, 2 * i + 1], jnp.cos(2 * jnp.pi / s[i]))
                seasonal_matrices.append(S)

            seasonal_matrix = reduce(lambda x, y: jnp.hstack((x, y)), seasonal_matrices)
            components.append(seasonal_matrix)

        # Combine components into a single state transition matrix
        F = jnp.block([[comp, jnp.zeros((comp.shape[0], sum([m.shape[1] for m in components]) - comp.shape[1]))] for comp in components])

        return F


    def build_observation_matrix(self):
        components = []

        # Trend component
        if self.trend_order is not None:
            components.append(jnp.eye(self.trend_order))

        # Seasonal component
        if self.seasonal_periods is not None:
            for s in self.seasonal_periods:
                components.append(jnp.eye(len(s) * 2))

        H = jnp.block([comp for comp in components])

        # If there's more than one time series, duplicate the observation matrix
        if self.n_series > 1:
            H = jnp.tile(H, (self.n_series, 1))

        # Add regressors to the observation matrix if provided
        if self.X is not None:
            H = jnp.hstack((H, self.X.T))

        return H

    def filter(self):
        raise NotImplementedError()

    def smooth(self):
        raise NotImplementedError()

    def forecast(self):
        raise NotImplementedError()
