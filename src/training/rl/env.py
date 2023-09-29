#  Copyright (c) 2023. Benjamin Schoofs

from typing import Optional

import torch
from tokenizers import Tokenizer

from src.data_engine.one_var_solve import Equation
from src.processing.tokenizer import format_decoding


class Env:
    """Environment for the EquationGame v[1var_int]"""
    _state: Optional[Equation]

    def __init__(self, tokenizer: Tokenizer):
        """Initialize the environment"""
        self._state = None
        self._tokenizer = tokenizer

    def state(self) -> Equation:
        """Get the current state"""
        return self._state

    def reset(self) -> torch.Tensor:
        """Reset the environment"""
        self._state = Equation.generate_rand_int_equation(10, -20, 10)
        str_repr = repr(self._state)
        return torch.tensor(self._tokenizer.encode(str_repr).ids)

    def apply_action(self, action: torch.Tensor):
        """Parse and apply the action"""
        str_repr: str = format_decoding(self._tokenizer.decode(action.tolist()))
        if "/add(" in str_repr:
            if "x)" in str_repr:
                argument = int(str_repr.replace("/add(", "").replace("x)", ""))
                self._state.add_coefficient(argument)
            else:
                argument = int(str_repr.replace("/add(", "").replace(")", ""))
                self._state.add_constant(argument)
        elif "/sub(" in str_repr:
            if "x)" in str_repr:
                argument = int(str_repr.replace("/sub(", "").replace("x)", ""))
                self._state.subtract_coefficient(argument)
            else:
                argument = int(str_repr.replace("/sub(", "").replace(")", ""))
                self._state.subtract_constant(argument)
        elif "/mul" in str_repr:
            argument = int(str_repr.replace("/mul(", "").replace(")", ""))
            self._state.multiply_constant(argument)
        elif "/div" in str_repr:
            argument = int(str_repr.replace("/div(", "").replace(")", ""))
            self._state.divide_constant(argument)
