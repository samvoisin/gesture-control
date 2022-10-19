# standard libraries
from queue import Queue
from typing import Any, Dict


class QueueError(Exception):
    pass


class VotingQueue(Queue):
    """
    A Queue where a popular vote is taken on labeled elements and the most popular label is popped from the queue.
    """

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize)
        self.vote_tally: Dict[int, int] = dict()
        self.curr_lab = None

    def put(self, item: Any):
        """
        Custom put implementation to avoid overfilling the queue.

        Args:
            item (Any): To be put in queue

        Raises:
            QueueError: _description_
        """
        if self.full():
            raise QueueError("Queue is full. Cannot put another element.")
        else:
            super().put(item)

    def vote(self) -> int:
        """
        Tally's popular vote of all entities in the queue. Items are removed from the queue in the process.

        Raises:
            QueueError: If `vote` is called when queue is empty

        Returns:
            int: Most popular label
        """
        self.vote_tally = dict()  # reset tally

        if self.empty():
            raise QueueError("Voting queue is empty.")

        while not self.empty():
            lab = self.get()
            lab_count = self.vote_tally.get(lab, 0)
            lab_count += 1
            self.vote_tally[lab] = lab_count

        pop_vote = None
        pop_vote_ct = 0
        for k, v in self.vote_tally.items():
            if v > pop_vote_ct:
                pop_vote = k
        return pop_vote
