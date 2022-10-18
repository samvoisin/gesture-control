# standard libraries
from typing import List, Optional

# external libraries
import pytest

# gestrol library
from gestrol.voting_queue import VotingQueue


def test_empty_voting_behavior():
    """
    Test error raised when vote is called on empty queue.
    """
    vq = VotingQueue(maxsize=5)
    with pytest.raises(ValueError):
        vq.vote()


@pytest.mark.parametrize(["labels", "exp_res"], [([6, 5, 4], 4), ([1, 2, 3, 3, 3], 3)])
def test_count_vote(labels: List[int], exp_res: Optional[int]):
    """
    Test vote counting method.
    """
    vq = VotingQueue(maxsize=5)
    print(labels)
    for label in labels:
        vq.put(label)

    assert vq.vote() == exp_res
