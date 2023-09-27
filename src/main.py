#  Copyright (c) 2023. Benjamin Schoofs

import torch

from src.training.one_var_int_loop import one_var_int_loop

from accelerate import notebook_launcher


def main():
    # set seed
    torch.manual_seed(0)
    # run
    notebook_launcher(one_var_int_loop, num_processes=1)


if __name__ == "__main__":
    main()
