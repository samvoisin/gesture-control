# standard libraries
from typing import List, Optional

# external libraries
import pytest

# gestrol library
from gesturemote.voting_queue import PopularVoteQueue, QueueError


def test_empty_voting_behavior():
    """
    Test error raised when vote is called on empty queue.
    """
    vq = PopularVoteQueue(maxsize=5)
    with pytest.raises(QueueError):
        vq.vote()


def test_overfill_behavior():
    """
    Test for error when overfilling queue.
    """
    vq = PopularVoteQueue(maxsize=5)
    labels = [1, 2, 3, 3, 3]
    for label in labels:
        vq.put(label)

    with pytest.raises(QueueError):
        vq.put(0)


@pytest.mark.parametrize(["labels", "exp_res"], [([6, 5, 4], 6), ([1, 2, 3, 3, 3], 3), ([1, 3, 3, 3, 1], 3)])
def test_count_vote(labels: List[int], exp_res: Optional[int]):
    """
    Test vote counting method.
    """
    vq = PopularVoteQueue(maxsize=5)
    print(labels)
    for label in labels:
        vq.put(label)

    assert vq.vote() == exp_res
