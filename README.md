
# GestuReMote

## What is it?

GestuReMote (pronounced "gesture-mote") is a tool for interfacing with a computer through physical gestures captured via webcam.

## Why use it?

Gesture control interfaces provide several advantages over traditional computer interfaces.

* Natural and Intuitive Interaction: One of the key advantages of gesture-based interfaces is their innate intuitiveness. Physical gestures can provide a direct mental link between user and object - for example, pointing with an index finger. Control through a tradition mouse or trackpad is indirect.
* Physical and Cognitive Engagement: Gesture-based interfaces require users to engage both physically and cognitively. This engagement can enhance user immersion and involvement with the tasks at hand.
* Accessibility: Gesture-based interfaces have the potential to cater to a broader range of users, including those with physical disabilities or impairments.

## How does it work?

### TL;DR

GestuReMote works by accessing a webcam attached to the machine, detecting hands and classifying gestures in the video stream, and executing associated commands.

### The details

GestuReMote is implemented in three layers.
1. The input layer: Images are streamed by device's camera and passed to layer 2.
2. The model layer: Images are preprocessed and passed into an object detection model where hands are identified and gestures are classified.
3. The control layer: Gesture classifications are interpreted as gesture routines and control routines are performed.

GestuReMote relies on an SSDLite model with a MobileNetV3 backbone trained on the [HaGRID](https://arxiv.org/abs/2206.08219) (HAnd Gesture Recognition Image Dataset). The model weights can be downloaded [here](https://github.com/hukenovs/hagrid/tree/master). Currently this library supports `SSDLiteMobileNetV3Small`. Thanks to  the HaGRID team for the hard work.

Supported gestures and their functions are limited currently as development is early. See `macos_routines.py` for a few simple gesture routines implemented for MacOS. These routines are illustrative in nature. Improved routines are in development.


