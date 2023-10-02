#  Copyright (c) 2023. Benjamin Schoofs

import torch

from src.training.one_var_int_loop import one_var_int_loop


def main():
    # set seed
    torch.manual_seed(0)
    # run
    one_var_int_loop()


if __name__ == "__main__":
    main()
