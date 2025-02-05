from __future__ import annotations

from typing import override

import jax.numpy as jnp
from jax import tree
from tjax import JaxRealArray, leaky_integrate
from tjax.dataclasses import dataclass
from tjax.gradient import GradientState, GradientTransformation

__all__ = ['MomentumGradient', 'MomentumState']


@dataclass
class MomentumState[Variables](GradientState):
    momentum: Variables


@dataclass
class MomentumGradient[Variables](GradientTransformation[MomentumState[Variables], Variables]):
    time_step: JaxRealArray
    friction: JaxRealArray

    @override
    def init(self, parameters: Variables) -> MomentumState[Variables]:
        z = tree.map(jnp.zeros_like, parameters)
        return MomentumState[Variables](z)

    @override
    def update(self,
               gradient: Variables,
               state: MomentumState[Variables],
               parameters: Variables | None
               ) -> tuple[Variables, MomentumState[Variables]]:
        # Calculate gradient.
        negative_gradient = tree.map(jnp.negative, gradient)
        new_momentum = self._leaky_integrate(state.momentum, negative_gradient)
        gradient = tree.map(lambda x: x * self.time_step, new_momentum)
        return gradient, MomentumState[Variables](new_momentum)

    def _leaky_integrate(self, value: Variables, drift: Variables) -> Variables:
        return tree.map(  # type: ignore[no-any-return]
            lambda x, y: leaky_integrate(x, self.time_step, y, self.friction),
            value, drift)
