from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import override

import jax.numpy as jnp
from jax import tree
from tjax import RealNumeric, leaky_integrate
from tjax.dataclasses import dataclass
from tjax.gradient import ThirdOrderGradientTransformation

from .harmonic import HarmonicState

__all__ = ['RelativisticGradient']


@dataclass
class RelativisticGradient[Variables](ThirdOrderGradientTransformation[HarmonicState[Variables],
                                                                       Variables]):
    time_step: RealNumeric
    friction_learning_rate: RealNumeric = 1e-2
    minimum_log_friction: RealNumeric = -10.0
    maximum_log_friction: RealNumeric = 10.0

    @override
    def init(self, parameters: Variables) -> HarmonicState[Variables]:
        z = tree.map(jnp.zeros_like, parameters)
        return HarmonicState[Variables](z, z, z, z)

    @override
    def third_order_update(self,
                           gradient: Variables,
                           state: HarmonicState[Variables],
                           parameters: Variables | None,
                           hessian_vector_product: Callable[[Variables], Variables],
                           hessian_diagonal: Variables
                           ) -> tuple[Variables, HarmonicState[Variables]]:
        friction = tree.map(jnp.exp, state.log_friction)
        time_step = tree.map(lambda x: x * self.time_step, hessian_diagonal)

        # Calculate gradient.
        negative_gradient = tree.map(jnp.negative, gradient)
        new_momentum = self._leaky_integrate(state.momentum, negative_gradient, friction, time_step)
        gradient = tree.map(jnp.multiply, time_step, new_momentum)

        # Update v.
        v_dot_dot = tree.map(jnp.negative, hessian_vector_product(state.v))
        new_v_dot = self._leaky_integrate(state.v_dot,
                                          tree.map(lambda x, y, z: x - z * y,
                                                   v_dot_dot,
                                                   new_momentum,
                                                   friction),
                                          friction,
                                          time_step)
        new_v = self._integrate(state.v, new_v_dot, time_step)

        # Update friction.
        log_friction_dot = tree.map(lambda x, y: self.friction_learning_rate * x * y,
                                    negative_gradient, new_v)
        new_log_friction = self._integrate(state.log_friction, log_friction_dot, time_step)
        new_log_friction = tree.map(partial(jnp.clip,
                                            a_min=self.minimum_log_friction,
                                            a_max=self.maximum_log_friction),
                                    new_log_friction)

        return gradient, HarmonicState[Variables](new_momentum, new_log_friction, new_v, new_v_dot)

    def _integrate(self, value: Variables, drift: Variables, time_step: Variables) -> Variables:
        return tree.map(  # type: ignore[no-any-return]
            lambda x, y, t: leaky_integrate(x, t, y, None), value, drift, time_step)

    def _leaky_integrate(self, value: Variables, drift: Variables, friction: Variables,
                         time_step: Variables) -> Variables:
        return tree.map(  # type: ignore[no-any-return]
            lambda x, y, z, t: leaky_integrate(x, t, y, z),
            value, drift, friction, time_step)
