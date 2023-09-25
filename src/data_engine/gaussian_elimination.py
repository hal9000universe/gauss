#  Copyright (c) 2023. Benjamin Schoofs

from torch import Tensor, no_grad
from typing import Tuple

from src.data_engine.row_ops import SwapRows, MultiplyRow, ReduceRows, Record, Quotient


# elementary row operations
def swap_rows(system: Tensor, i: int, j: int):
    """Swaps rows i and j of the system of equations.
    
    Args:
        system (Tensor): system of equations
        i (int): row index
        j (int): row index
    """
    system[[i, j]] = system[[j, i]]


def multiply_row(system: Tensor, i: int, k: float):
    """Multiplies row i of the system of equations by k.
    
    Args:
        system (Tensor): system of equations
        i (int): row index
        k (float): scalar
    """
    system[i] *= k


def reduce_rows(system: Tensor, reduced_row_idx: int, other_row_idx: int, k: float):
    """Adds k times row j to row i of the system of equations.
    
    Args:
        system (Tensor): system of equations
        reduced_row_idx (int): row index of row to be reduced
        other_row_idx (int): row index of row to be added
        k (float): scalar
    """
    system[reduced_row_idx] += k * system[other_row_idx]


@no_grad()
def gaussian_elimination(system: Tensor) -> Tuple[Tensor, Record]:
    """Transforms the system of equations to reduced row echelon form 
    by employing elementary row operations.
    
    Args:
        system (Tensor): system of equations
        
    Returns:
        Tensor: system of equations in reduced row echelon form
    """
    num_variables: int = system.shape[1] - 1
    num_equations: int = system.shape[0]

    record: Record = Record()

    # forward elimination
    for k in range(0, num_variables):

        # initialize maximum value and index for pivot
        max_value: float = system[k, k]
        max_idx: int = k

        # find pivot
        for i in range(k + 1, num_equations):
            if abs(system[i, k]) > abs(max_value):
                max_value = system[i, k]
                max_idx = i

        # avoid division by zero
        if abs(max_value) < 1e-10:
            continue

        # swap rows
        if max_idx != k:
            record.add(system.clone().detach(), SwapRows(k, max_idx))
            swap_rows(system, k, max_idx)

        # normalize pivot rows
        record.add(system.clone().detach(), MultiplyRow(k, Quotient(1., system[k, k].item())))
        multiply_row(system, k, 1. / system[k, k])

        # reduce rows
        for i in range(k + 1, num_equations):
            record.add(system.clone().detach(), ReduceRows(i, k, Quotient(-system[i, k].item(),
                                                                          system[k, k].item())))
            factor: float = system[i, k] / system[k, k]
            reduce_rows(system, i, k, -factor)

    # backward elimination
    for k in range(num_variables - 1, 0, -1):
        for i in range(k - 1, -1, -1):
            record.add(system.clone().detach(), ReduceRows(i, k, Quotient(-system[i, k].item(), system[k, k].item())))
            factor: float = system[i, k] / system[k, k]
            reduce_rows(system, i, k, -factor)

    return system, record
