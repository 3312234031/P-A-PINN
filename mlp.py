
from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.utils import initializer

class MLP(base.Arch):

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        hidden_size: Union[int, Tuple[int, ...]],
        activation: str = "tanh",
        skip_connection: bool = False,
        weight_norm: bool = False,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        periods: Optional[Dict[int, Tuple[float, bool]]] = None,
        fourier: Optional[Dict[str, Union[float, int]]] = None,
        random_weight: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.linears = []
        self.acts = []
        self.periods = periods
        self.fourier = fourier
        if periods:
            self.period_emb = PeriodEmbedding(periods)

        if isinstance(hidden_size, (tuple, list)):
            if num_layers is not None:
                raise ValueError(
                    "num_layers should be None when hidden_size is specified"
                )
        elif isinstance(hidden_size, int):
            if not isinstance(num_layers, int):
                raise ValueError(
                    "num_layers should be an int when hidden_size is an int"
                )
            hidden_size = [hidden_size] * num_layers
        else:
            raise ValueError(
                f"hidden_size should be list of int or int, but got {type(hidden_size)}"
            )

        # initialize FC layer(s)
        cur_size = len(self.input_keys) if input_dim is None else input_dim
        if input_dim is None and periods:
            # period embeded channel(s) will be doubled automatically
            # if input_dim is not specified
            cur_size += len(periods)

        if fourier:
            self.fourier_emb = FourierEmbedding(
                cur_size, fourier["dim"], fourier["scale"]
            )
            cur_size = fourier["dim"]

        for i, _size in enumerate(hidden_size):
            if weight_norm:
                self.linears.append(WeightNormLinear(cur_size, _size))
            elif random_weight:
                self.linears.append(
                    RandomWeightFactorization(
                        cur_size,
                        _size,
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                )
            else:
                self.linears.append(nn.Linear(cur_size, _size))

            # initialize activation function
            self.acts.append(
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(_size)
            )
            # special initialization for certain activation
            # TODO: Adapt code below to a more elegant style
            if activation == "siren":
                if i == 0:
                    act_mod.Siren.init_for_first_layer(self.linears[-1])
                else:
                    act_mod.Siren.init_for_hidden_layer(self.linears[-1])

            cur_size = _size

        self.linears = nn.LayerList(self.linears)
        self.acts = nn.LayerList(self.acts)
        if random_weight:
            self.last_fc = RandomWeightFactorization(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        else:
            self.last_fc = nn.Linear(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
            )

        self.skip_connection = skip_connection

    def forward_tensor(self, x):
        y = x
        skip = None
        for i, linear in enumerate(self.linears):
            y = linear(y)
            if self.skip_connection and i % 2 == 0:
                if skip is not None:
                    skip = y
                    y = y + skip
                else:
                    skip = y
            y = self.acts[i](y)

        y = self.last_fc(y)

        return y

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        if self.periods:
            x = self.period_emb(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)

        if self.fourier:
            y = self.fourier_emb(y)

        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y


class ModifiedMLP(base.Arch):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        hidden_size: int,
        activation: str = "tanh",
        skip_connection: bool = False,
        weight_norm: bool = False,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        periods: Optional[Dict[int, Tuple[float, bool]]] = None,
        fourier: Optional[Dict[str, Union[float, int]]] = None,
        random_weight: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.linears = []
        self.acts = []
        self.periods = periods
        self.fourier = fourier
        if periods:
            self.period_emb = PeriodEmbedding(periods)
        if isinstance(hidden_size, int):
            if not isinstance(num_layers, int):
                raise ValueError("num_layers should be an int")
            hidden_size = [hidden_size] * num_layers
        else:
            raise ValueError(f"hidden_size should be int, but got {type(hidden_size)}")

        # initialize FC layer(s)
        cur_size = len(self.input_keys) if input_dim is None else input_dim
        if input_dim is None and periods:
            # period embeded channel(s) will be doubled automatically
            # if input_dim is not specified
            cur_size += len(periods)

        if fourier:
            self.fourier_emb = FourierEmbedding(
                cur_size, fourier["dim"], fourier["scale"]
            )
            cur_size = fourier["dim"]

        self.embed_u = nn.Sequential(
            (
                WeightNormLinear(cur_size, hidden_size[0])
                if weight_norm
                else (
                    nn.Linear(cur_size, hidden_size[0])
                    if random_weight is None
                    else RandomWeightFactorization(
                        cur_size,
                        hidden_size[0],
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                )
            ),
            (
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(hidden_size[0])
            ),
        )
        self.embed_v = nn.Sequential(
            (
                WeightNormLinear(cur_size, hidden_size[0])
                if weight_norm
                else (
                    nn.Linear(cur_size, hidden_size[0])
                    if random_weight is None
                    else RandomWeightFactorization(
                        cur_size,
                        hidden_size[0],
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                )
            ),
            (
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(hidden_size[0])
            ),
        )

        for i, _size in enumerate(hidden_size):
            if weight_norm:
                self.linears.append(WeightNormLinear(cur_size, _size))
            elif random_weight:
                self.linears.append(
                    RandomWeightFactorization(
                        cur_size,
                        _size,
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                )
            else:
                self.linears.append(nn.Linear(cur_size, _size))

            # initialize activation function
            self.acts.append(
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(_size)
            )
            # special initialization for certain activation
            # TODO: Adapt code below to a more elegant style
            if activation == "siren":
                if i == 0:
                    act_mod.Siren.init_for_first_layer(self.linears[-1])
                else:
                    act_mod.Siren.init_for_hidden_layer(self.linears[-1])

            cur_size = _size

        self.linears = nn.LayerList(self.linears)
        self.acts = nn.LayerList(self.acts)
        if random_weight:
            self.last_fc = RandomWeightFactorization(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        else:
            self.last_fc = nn.Linear(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
            )

        self.skip_connection = skip_connection

    def forward_tensor(self, x):
        u = self.embed_u(x)
        v = self.embed_v(x)

        y = x
        skip = None
        for i, linear in enumerate(self.linears):
            y = linear(y)
            y = self.acts[i](y)
            y = y * u + (1 - y) * v
            if self.skip_connection and i % 2 == 0:
                if skip is not None:
                    skip = y
                    y = y + skip
                else:
                    skip = y

        y = self.last_fc(y)

        return y

    def forward(self, x):
        x_identity = x
        if self._input_transform is not None:
            x = self._input_transform(x)

        if self.periods:
            x = self.period_emb(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)

        if self.fourier:
            y = self.fourier_emb(y)

        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x_identity, y)
        return y
