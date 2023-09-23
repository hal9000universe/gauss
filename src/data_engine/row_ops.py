#  Copyright (c) 2023. Benjamin Schoofs

import torch
from typing import List, NamedTuple, Tuple


class Quotient:
    """Represents a quotient."""
    _value: float
    _numerator: float
    _denominator: float

    def __init__(self, numerator: float, denominator: float):
        """Initializes the quotient with the numerator and the denominator.
        
        Args:
            numerator (float): numerator
            denominator (float): denominator
        """
        self._value = numerator / denominator
        self._numerator = numerator
        self._denominator = denominator

    def value(self) -> float:
        """Returns the value.

        Returns:
            float: value
        """
        return self._value

    def numerator_denominator(self) -> Tuple[float, float]:
        """Returns the numerator and the denominator.
        
        Returns:
            Tuple[float, 2]: numerator and denominator
        """
        return self._numerator, self._denominator

    def __repr__(self):
        """Returns a string representation of the quotient.
        
        Returns:
            str: string representation of the quotient
        """
        return f"/div({self._numerator},{self._denominator})"


class Operation:
    """Represents an operation performed on a system of equations."""
    _name: str

    def __init__(self, name: str):
        """Initializes the operation with a name.
        
        Args:
            name (str): name of the operation
        """
        self._name = name


class SwapRows(Operation):
    """Represents the operation of swapping two rows of a system of equations."""
    _i: int
    _j: int

    def __init__(self, i: int, j: int):
        """Initializes the operation with the row indices of the rows to be swapped.
        
        Args:
            i (int): row index
            j (int): row index
        """
        super().__init__("/swap_rows")
        self._i = i
        self._j = j

    def __repr__(self):
        """Returns a string representation of the operation.
        
        Returns:
            str: string representation of the operation
        """
        return f"{self._name}({self._i},{self._j})"


class MultiplyRow(Operation):
    """Represents the operation of multiplying a row of a system of equations by a scalar."""
    _i: int
    _k: Quotient

    def __init__(self, i: int, k: Quotient):
        """Initializes the operation with the row index of the row to be multiplied and the scalar.
        
        Args:
            i (int): row index
            k (Quotient): scalar
        """
        super().__init__("/multiply_row")
        self._i = i
        self._k = k

    def __repr__(self):
        """Returns a string representation of the operation.
        
        Returns:
            str: string representation of the operation
        """
        return f"{self._name}({self._i},{self._k})"


class ReduceRows(Operation):
    """Represents the operation of adding a multiple of one row to another row of a system of equations."""
    _i: int
    _j: int
    _k: Quotient

    def __init__(self, i: int, j: int, k: Quotient):
        """Initializes the operation with the row indices of the rows to be added and the scalar.
        
        Args:
            i (int): row index
            j (int): row index
            k (Value): scalar
        """
        super().__init__("/reduce_rows")
        self._i = i
        self._j = j
        self._k = k

    def __repr__(self):
        """Returns a string representation of the operation.
        
        Returns:
            str: string representation of the operation
        """
        return f"{self._name}({self._i},{self._j},{self._k})"


class SystemOperationPair(NamedTuple):
    """Represents a pair of a system of equations and the corresponding operation."""
    system: torch.Tensor
    operation: Operation


class Record:
    """Records a history of system states and operations."""
    _system_op_pairs: List[SystemOperationPair]

    def __init__(self):
        """Initializes the record with the initial state of the system of equations."""
        self._system_op_pairs = []

    def add(self, system: torch.Tensor, operation: Operation):
        """Adds a pair of a system of equations and the corresponding operation to the record.
        
        Args:
            system (torch.Tensor): system of equations
            operation (str): operation performed on the system of equations
        """
        self._system_op_pairs.append(SystemOperationPair(system, operation))

    def system_op_pairs(self) -> List[SystemOperationPair]:
        """Returns the list of system-operation pairs.
        
        Returns:
            List[SystemOperationPair]: list of system-operation pairs
        """
        return self._system_op_pairs
