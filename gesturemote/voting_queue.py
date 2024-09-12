from collections import Counter
from queue import Queue
from typing import Any

from gesturemote.logger_config import configure_logger


class QueueError(Exception):
    """
    Custom exception for queue errors.
    """

    pass


class PopVoteCounter(Counter):
    """
    Calculate popular vote of a list of elements.
    """

    def popular_vote(self) -> Any:
        """
        Calculate popular vote.

        Returns:
            Most common item in the list.
        """
        pop_vote = self.most_common(1)
        return pop_vote[0][0]


class VoteQueue:
    """
    A Queue where a vote is taken on labeled elements and the most voted label is popped from the queue.
    """

    def __init__(self, maxsize: int) -> None:
        """
        Args:
            maxsize (int): Queue size.
        """
        self.queue: Queue = Queue(maxsize=maxsize)
        self.vote_counter = PopVoteCounter()
        self.logger = configure_logger()

    def __repr__(self) -> str:
        return str(self.queue.queue)

    def is_full(self) -> bool:
        """
        Check if the queue is full.

        Returns:
            bool: True if the queue is full, False otherwise.
        """
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
            self.logger.info("Voting queue status: %s", self)

    def vote(self) -> Any:
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
