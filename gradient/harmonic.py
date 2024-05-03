from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import override

import jax.numpy as jnp
from jax import tree
from tjax import JaxRealArray, leaky_integrate
from tjax.dataclasses import dataclass, field
from tjax.gradient import GradientState, SecondOrderGradientTransformation

__all__ = ['HarmonicGradient', 'HarmonicState']


@dataclass
class HarmonicState[Variables](GradientState):
    momentum: Variables
    log_friction: Variables
    v: Variables
    v_dot: Variables


@dataclass
class HarmonicGradient[Variables](SecondOrderGradientTransformation[HarmonicState[Variables],
                                                                    Variables]):
    time_step: JaxRealArray
    friction_learning_rate: JaxRealArray = field(default_factory=lambda: jnp.asarray(1e-2))
    minimum_log_friction: JaxRealArray = field(default_factory=lambda: jnp.asarray(-10.0))
    maximum_log_friction: JaxRealArray = field(default_factory=lambda: jnp.asarray(10.0))

    @override
    def init(self, parameters: Variables) -> HarmonicState[Variables]:
        z = tree.map(jnp.zeros_like, parameters)
        return HarmonicState[Variables](z, z, z, z)

    @override
    def second_order_update(self,
                            gradient: Variables,
                            state: HarmonicState[Variables],
                            parameters: None | Variables,
                            hessian_vector_product: Callable[[Variables], Variables]) -> (
                                tuple[Variables, HarmonicState[Variables]]):
        friction = tree.map(jnp.exp, state.log_friction)

        # Calculate gradient.
        negative_gradient = tree.map(jnp.negative, gradient)
        new_momentum = self._leaky_integrate(state.momentum, negative_gradient, friction)
        gradient = tree.map(lambda x: x * self.time_step, new_momentum)

        # Update v.
        v_dot_dot = tree.map(jnp.negative, hessian_vector_product(state.v))
        new_v_dot = self._leaky_integrate(state.v_dot,
                                          tree.map(lambda x, y, z: x - z * y,
                                                   v_dot_dot,
                                                   new_momentum,
                                                   friction),
                                          friction)
        new_v = self._integrate(state.v, new_v_dot)

        # Update friction.
        log_friction_dot = tree.map(lambda x, y: self.friction_learning_rate * x * y,
                                    negative_gradient, new_v)
        new_log_friction = self._integrate(state.log_friction, log_friction_dot)
        new_log_friction = tree.map(partial(jnp.clip,
                                            a_min=self.minimum_log_friction,
                                            a_max=self.maximum_log_friction),
                                    new_log_friction)

        return gradient, HarmonicState[Variables](new_momentum, new_log_friction, new_v, new_v_dot)

    def _integrate(self, value: Variables, drift: Variables) -> Variables:
        return tree.map(  # type: ignore[no-any-return]
            lambda x, y: leaky_integrate(x, self.time_step, y, None), value, drift)

    def _leaky_integrate(self, value: Variables, drift: Variables, friction: Variables
                         ) -> Variables:
        return tree.map(  # type: ignore[no-any-return]
            lambda x, y, z: leaky_integrate(x, self.time_step, y, z),
            value, drift, friction)
