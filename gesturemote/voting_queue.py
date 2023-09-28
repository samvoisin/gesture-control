# standard libraries
import logging
from collections import Counter
from queue import Queue
from typing import Any


class QueueError(Exception):
    pass


class PopVoteCounter(Counter):
    def popular_vote(self) -> int:
        pop_vote = self.most_common(1)
        return pop_vote[0][0]


class VoteQueue:
    """
    A Queue where a vote is taken on labeled elements and the most voted label is popped from the queue.
    """

    def __init__(self, maxsize: int, verbose: bool = False) -> None:
        self.queue: Queue = Queue(maxsize=maxsize)
        self.vote_counter = PopVoteCounter()
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)

    def __repr__(self) -> str:
        return str(self.queue.queue)

    def is_full(self) -> bool:
        return self.queue.full()

    def put(self, item: Any):
        """
        Custom put implementation to avoid overfilling the queue.

        Args:
            item (Any): To be put in queue

        Raises:
            QueueError: raised if voting queue is full.
        """
        if self.is_full():
            raise QueueError("Voting queue is full. Cannot put another element.")
        else:
            self.queue.put(item)
            self.logger.info(f"Voting queue status: {self}")

    def vote(self) -> int:
        """
        Tally popular vote of all entities in the queue. Items are removed from the queue in the process.

        Raises:
            QueueError: Raised if called when queue is empty

        Returns:
            int: most popular value
        """
        if self.queue.empty():
            raise QueueError("Voting queue is empty.")

        self.vote_counter.clear()

        self.logger.info("Voting on queue: %s", self.queue.queue)
        while not self.queue.empty():
            self.vote_counter[self.queue.get()] += 1

        return self.vote_counter.popular_vote()
