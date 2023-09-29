#  Copyright (c) 2023. Benjamin Schoofs

from typing import Optional

import torch
from tokenizers import Tokenizer

from src.data_engine.one_var_solve import Equation
from src.processing.tokenizer import format_decoding


class Env:
    """Environment for the EquationGame v[1var_int]"""
    _state: Optional[Equation]
    _num_steps: Optional[int]
    _max_steps: int
    _tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer):
        """Initialize the environment"""
        self._state = None
        self._num_steps = None
        self._max_steps = 6
        self._tokenizer = tokenizer

    def state(self) -> Equation:
        """Get the current state"""
        return self._state

    def set_state(self, state):
        """Set the current state"""
        self._state = state

    def reset(self) -> torch.Tensor:
        """Reset the environment"""
        self._state = Equation.generate_rand_int_equation(10, -20, 10)
        self._num_steps = 0
        str_repr = repr(self._state)
        return torch.tensor(self._tokenizer.encode(str_repr).ids)

    def _apply_action(self, action: torch.Tensor):
        """Parse and apply the action

        Args:
            action (torch.Tensor): The action to apply (as token ids)
        """
        self._num_steps += 1
        for i, tok_id in enumerate(action):
            if tok_id == self._tokenizer.token_to_id("/add("):
                number = ""
                for action_id in action[i + 1:]:
                    if action_id == self._tokenizer.token_to_id("x"):
                        self._state.add_coefficient(float(number))
                        return
                    elif action_id == self._tokenizer.token_to_id("x)"):
                        self._state.add_coefficient(float(number))
                        return
                    elif action_id == self._tokenizer.token_to_id(")"):
                        self._state.add_constant(float(number))
                        return
                    else:
                        number += self._tokenizer.id_to_token(action_id)
            elif tok_id == self._tokenizer.token_to_id("/sub("):
                number = ""
                for action_id in action[i + 1:]:
                    if action_id == self._tokenizer.token_to_id("x"):
                        self._state.subtract_coefficient(float(number))
                        return
                    elif action_id == self._tokenizer.token_to_id("x)"):
                        self._state.subtract_coefficient(float(number))
                        return
                    elif action_id == self._tokenizer.token_to_id(")"):
                        self._state.subtract_constant(float(number))
                        return
                    else:
                        number += self._tokenizer.id_to_token(action_id)
            elif tok_id == self._tokenizer.token_to_id("/mul("):
                number = ""
                for action_id in action[i + 1:]:
                    if action_id == self._tokenizer.token_to_id(")"):
                        self._state.multiply_constant(float(number))
                        return
                    else:
                        number += self._tokenizer.id_to_token(action_id)
            elif tok_id == self._tokenizer.token_to_id("/div("):
                number = ""
                for action_id in action[i + 1:]:
                    if action_id == self._tokenizer.token_to_id(")"):
                        self._state.divide_constant(float(number))
                        return
                    else:
                        number += self._tokenizer.id_to_token(action_id)
            elif tok_id == self._tokenizer.token_to_id("/add(-"):
                number = "-"
                for action_id in action[i + 1:]:
                    if action_id == self._tokenizer.token_to_id("x"):
                        self._state.add_coefficient(float(number))
                        return
                    elif action_id == self._tokenizer.token_to_id("x)"):
                        self._state.add_coefficient(float(number))
                        return
                    elif action_id == self._tokenizer.token_to_id(")"):
                        self._state.add_constant(float(number))
                        return
                    else:
                        number += self._tokenizer.id_to_token(action_id)
            elif tok_id == self._tokenizer.token_to_id("/sub(-"):
                number = "-"
                for action_id in action[i + 1:]:
                    if action_id == self._tokenizer.token_to_id("x"):
                        self._state.subtract_coefficient(float(number))
                        return
                    elif action_id == self._tokenizer.token_to_id("x)"):
                        self._state.subtract_coefficient(float(number))
                        return
                    elif action_id == self._tokenizer.token_to_id(")"):
                        self._state.subtract_constant(float(number))
                        return
                    else:
                        number += self._tokenizer.id_to_token(action_id)
            elif tok_id == self._tokenizer.token_to_id("/mul(-"):
                number = "-"
                for action_id in action[i + 1:]:
                    if action_id == self._tokenizer.token_to_id(")"):
                        self._state.multiply_constant(float(number))
                        return
                    else:
                        number += self._tokenizer.id_to_token(action_id)
            elif tok_id == self._tokenizer.token_to_id("/div(-"):
                number = "-"
                for action_id in action[i + 1:]:
                    if action_id == self._tokenizer.token_to_id(")"):
                        self._state.divide_constant(float(number))
                        return
                    else:
                        number += self._tokenizer.id_to_token(action_id)

    def _is_done(self) -> bool:
        """Check if the environment is done

        Returns:
            bool: True if the environment is done, False otherwise
        """
        return self._state.is_solved() or self._num_steps > self._max_steps

    def _reward(self) -> float:
        """Get the reward for the current state

        Returns:
            float: The reward
        """
        if self._state.is_solved():
            return 1
        elif self._num_steps > self._max_steps:
            return -1
        else:
            return 0

    def step(self, action: torch.Tensor):
        """Apply the action and return the reward

        Args:
            action (torch.Tensor): The action to apply (as token ids)

        Returns:
            torch.Tensor: The reward for the action
        """
        self._apply_action(action)
        next_obs = self._tokenizer.encode(repr(self._state)).ids
        reward = self._reward()
        done = self._is_done()
        return next_obs, reward, done


if __name__ == '__main__':
    env_tokenizer = Tokenizer.from_file("tokenizer/one_var_tokenizer.json")

    env = Env(env_tokenizer)

    from src.models.transformer import MathFormer

    model = MathFormer(64, 128, 4, 2, env_tokenizer)

    obs = env.reset()

    action = model.generate_action(obs.view(1, -1))

    print(format_decoding(env_tokenizer.decode(action.tolist())), action)

    print(env.state())

    env.step(action)

    print(env.state())
