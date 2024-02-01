

# GestuReMote

![macos build workflow](https://github.com/samvoisin/gesture-control/actions/workflows/build-macos.yaml/badge.svg?event=push)
[![codecov](https://codecov.io/gh/samvoisin/gesture-control/graph/badge.svg?token=K16WHE65U4)](https://codecov.io/gh/samvoisin/gesture-control)

## What is it?

GestuReMote (pronounced "gesture-mote") is a tool for interfacing with a computer through physical gestures captured via optical sensors. Right now this means a standard webcam - though I plan to construct a dedicated device in the future.

Currently GestuReMote supports systems running MacOS. Support for Linux operating systems and Windows will be here in the next few months.

This is an open source project that work on when I am able, so please be patient. If you would like to get involved in the development of GestuReMote don't hesitate to reach out!

## Why use it?

Gesture control interfaces provide several advantages over traditional computer interfaces.

* Intuitive control: One of the key advantages of gesture-based interfaces is their innate intuitiveness. Physical gestures can provide a direct link between user and machine. This is in contrast to indirect control provided by a traditional mouse or trackpad. As an illustration, consider the scenario in which you want to access a particular application on your machine. Simply pointing at the application icon with your index finger is a very intuitive method of control. Moving a mouse which in turn moves the cursor by some dpi ratio is a much less direct way to accomplish the same task.
* Physical and cognitive alignment: Gesture-based interfaces require users to engage both physically and cognitively. This engagement can enhance user immersion and involvement with the tasks being performed.
* Accessibility: Gesture-based interfaces have the potential to cater to a broader range of users, including those with a mobility or physical disability.

## How does it work?

### TL;DR

GestuReMote works by accessing a webcam attached to the machine, detecting hands and classifying gestures in the video stream, and executing associated commands.

### The details

GestuReMote currently works by accessing a local webcam via [OpenCV](https://opencv.org/). Images captured by the webcam are then processed with [Google's mediapipe library](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer) to identify gestures and landmarks on the hand. Finally, the identified landmarks and gestures are used to change the cursor's position, right and left click, and perform some common user-interface actions.

Currently only MacOS is supported. Support for Windows and Linux operating systems is coming as soon as possible!

### Installing

First, download the most recent [HandGestureClassifier task](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer#models). Save the `.task` file to the `models` directory.

You can install GestuReMote with the `make init` recipe (see the `Makefile` in the repo's root for details). Alternatively, you can install this library using `pip install .` in the repository root.

### Using GestuReMote

There is a CLI available to make using GestuReMote as easy as possible. Activate GestuReMote through the CLI with:

```
gesturemote activate
```

You can use `gesturemote activate --help` for details on which options are available to you. If you are having a hard time getting GestuReMote to work, use the `--video-preview` option to see what the camera sees.

Once GestuReMote is active, it will wait for you to make the control gesture before doing anything.

* The control gesture is a closed fist. When you perform this gesture, GestuReMote will enter control mode. When you perform it again, control mode will be turned off.
* While gesturemote is in control mode, the cursor will track your index finger while it is in front of the camera.
* You can perform a primary click by touching your middle finger to your thumb. The primary click will remain engaged as long as your middle finger and thumb are touching. This allows click-and-drag behavior.
* You can perform a secondary click by touching your ring finger to your thumb.
* You can scroll up and down by putting your index and middle fingers together in parallel and moving them up and down. Moving them upwards relative to the center of the camera view sends the scroll-up signal. Moving them downwards relative to the center of the camera view sends the scroll-down signal. The scroll speed increases with the distance relative to the center of the camera view.
* You can use the "thumbs-up" and "thumbs-down" gestures to perform PageUp and PageDown actions respectively.