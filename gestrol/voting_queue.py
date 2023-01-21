# standard libraries
import logging
from collections import Counter
from queue import Queue


class QueueError(Exception):
    pass


class PopVoteCounter(Counter):
    def popular_vote(self) -> int:
        pop_vote = self.most_common(1)
        return pop_vote[0][0]


class PopularVoteQueue:
    """
    A Queue where a popular vote is taken on labeled elements and the most popular label is popped from the queue.
    """

    def __init__(self, maxsize: int) -> None:
        self.queue: Queue = Queue(maxsize=maxsize)
        self.vote_counter = PopVoteCounter()

    def __repr__(self) -> str:
        return str(self.queue.queue)

    def put(self, item: int):
        """
        Custom put implementation to avoid overfilling the queue.

        Args:
            item (Any): To be put in queue

        Raises:
            QueueError: raised if voting queue is full.
        """
        if self.queue.full():
            raise QueueError("Voting queue is full. Cannot put another element.")
        else:
            self.queue.put(item)

    def vote(self) -> int:
        """
        Tally popular vote of all entities in the queue. Items are removed from the queue in the process.

        Raises:
            QueueError: Raised if called when queue is empty

        Returns:
            int: most popular value
        """
        self.vote_counter.clear()

        if self.queue.empty():
            raise QueueError("Voting queue is empty.")

        logging.info("Voting on queue: %s", self.queue.queue)
        while not self.queue.empty():
            self.vote_counter[self.queue.get()] += 1

        return self.vote_counter.popular_vote()
