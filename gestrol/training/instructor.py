# standard libraries
from pathlib import Path
from random import shuffle
from time import sleep
from typing import Any, Dict, List, Protocol, Sequence

# external libraries
from yaml import safe_load


def _clear_prompt():
    print(" " * 50, end="\r")


class TrainingCameraInterface(Protocol):
    def __init__(self, **kwargs: Any):
        ...

    def __del__(self, **kwargs: Any):
        ...

    def get_frame(self, **kwargs: Any) -> Any:
        ...

    def record_video(self, **kwargs: Any):
        ...


class TrainingInstructor:
    """
    Capture gesture training data video clips
    """

    def __init__(self, training_script: Path, camera_interface: TrainingCameraInterface) -> None:
        self.cam = camera_interface
        with open(training_script, "r") as fh:
            instructions = safe_load(fh)
        self._parse_instructions_yaml(instructions)

    def _make_sample_queue(
        self, gesture_labels: Sequence[str], n_iters: int, shuffle_samples: bool = True
    ) -> List[str]:
        sample_queue = []
        for glabel in gesture_labels:
            sample_queue.extend([glabel] * n_iters)

        if shuffle_samples:
            shuffle(sample_queue)

        return sample_queue

    def _parse_instructions_yaml(self, instructions: Dict[str, Any]):
        self.training_data_dir = Path(instructions["training_data_dir"]).resolve()
        sample_size = instructions["sample_size"]
        self._sample_time = instructions["sample_time"]
        self._gesture_instructions = instructions["gesture_instructions"]
        self._sample_queue = self._make_sample_queue(self._gesture_instructions.keys(), n_iters=sample_size)
        self._sample_counts = {gesture_label: 0 for gesture_label in self._gesture_instructions.keys()}

    def record_sample(self, sample_label: str, video_format: str = ".mp4"):
        save_path = self.training_data_dir / sample_label / (str(self._sample_counts[sample_label]) + video_format)

        instruction = self._gesture_instructions[sample_label]
        print(
            f"Perform the following gesture when recording begins:\n{instruction}\nRecording will last for"
            f" {self._sample_time}s",
            end="\r",
        )
        sleep(5)

        for i in range(3, 0, -1):
            print(f"Recording to begin in {i}...", end="\r")
            sleep(1)

        _clear_prompt()
        print("Now recording...", end="\r")
        self.cam.record_video(save_path=save_path, vlen=self._sample_time)

    def collect_training_data(self):
        for sample_label in self._sample_queue:
            self.record_sample(sample_label)
            # wait for user to press a key
            print("Return hand to neutral position.", end="\r")
            sleep(1)
            _clear_prompt()
            for i in range(3, 0, -1):
                print(f"Next gesture in {i}...", end="\r")
                sleep(1)
            _clear_prompt()
