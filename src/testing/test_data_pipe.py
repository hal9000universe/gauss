#  Copyright (c) 2023. Benjamin Schoofs

import os

from src.data_engine.data_pipe import gen_data


def test_data_gen():
    """Checks whether the data is generated correctly."""
    # generate data
    gen_data()
    # check whether file exists
    assert os.path.exists("data/examples.txt")

    # check if every line contains ?
    # check if every line contains /n
    # check if every line contains /swap_rows or reduce_row or multiply_row
    with open("data/examples.txt", "r") as f:
        for line in f.readlines():
            assert "?" in line, "No ? seperator found."
            assert "/n" in line, "No /n seperator found."
            assert "/swap_rows" in line or "/reduce_rows" in line or "/multiply_row" in line, "No operation found."
