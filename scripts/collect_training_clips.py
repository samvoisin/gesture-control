# standard libraries
from pathlib import Path

# gestrol library
from gestrol.camera import OpenCVCameraInterface
from gestrol.training.instructor import TrainingInstructor


def main():
    """
    main program body. Intended to be run from repository root dir
    """
    training_instr_path = Path("./gestrol/training/training_instructions.yaml").resolve()
    cam = OpenCVCameraInterface()
    train_instr = TrainingInstructor(training_script=training_instr_path, camera_interface=cam)
    train_instr.collect_training_data()


if __name__ == "__main__":
    main()
