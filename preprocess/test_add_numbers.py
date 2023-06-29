from game_transcript import add_numbers
import pytest

def test_add_numbers(): #the idea is that you create a function to test if the function imported works as it should, considering all possible inputs
    assert add_numbers(2, 3) == 5
    assert add_numbers(10, -5) == 5
    assert add_numbers(0, 0) == 0