# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pose_engine import PoseEngine
import fluidsynth
import itertools
import time
"""
Trying to print the image to the screen...NOPE...

    if args.output:
      draw_objects(ImageDraw.Draw(pil_image), poses)
      pil_image.save(args.output)
      pil_image.show()



    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output',
                        help='File path for the result image with annotations')
    args = parser.parse_args()
"""

"""
PIL seems like a great resource for image processing. See pillow.readthedocs.io/en/3.1.x/reference/Image.html

Currently, pil_image.show() doesn't seem to be showing anything on the screen. Not sure if that is the way we're calling it, the coral dev board, or if it's the function .show() thats not working.
"""
"""
parse the output argument...maybe this will work to get the image on the screen.
I found these gems of code at ~/coral/tflite/python/examples/detection/detect_image.py
"""

OCTAVE = 12
FIFTH = 7
MINOR_THIRD = 3
CIRCLE_OF_FIFTHS = tuple((i * FIFTH) % OCTAVE
                         for i in range(OCTAVE))

# first 5 fifths in order, e.g. C G D A E => C D E G A
MAJOR_PENTATONIC = tuple(sorted(CIRCLE_OF_FIFTHS[:5]))

# same as pentatonic major but starting at 9
# e.g. C D E G A = 0 2 4 7 9 => 3 5 7 10 0 = C D E G A => A C D E G
MINOR_PENTATONIC = tuple(sorted((i + MINOR_THIRD) % OCTAVE
                                for i in MAJOR_PENTATONIC))

SCALE = MAJOR_PENTATONIC

# General Midi ids
OVERDRIVEN_GUITAR = 30
ELECTRIC_BASS_FINGER = 34
VOICE_OOHS = 54

CHANNELS = (OVERDRIVEN_GUITAR, ELECTRIC_BASS_FINGER, VOICE_OOHS)


class Identity:
    def __init__(self, color, base_note, instrument, extent=2*OCTAVE):
        self.color = color
        self.base_note = base_note
        self.channel = CHANNELS.index(instrument)
        self.extent = extent


IDENTITIES = (
    Identity('cyan', 24, OVERDRIVEN_GUITAR),
    Identity('magenta', 12, ELECTRIC_BASS_FINGER),
    Identity('yellow', 36, VOICE_OOHS),
)


def draw_objects(draw,objs):
    '''draws the bounding box and label for each object.'''
    for obj in objs:
        for label, bbox in obj.keypoints.items():
            draw.rectangle([(bbox.yx[1], bbox.yx[0]), (bbox.yx[1], bbox.yx[0])],
                            outline='red')
            draw.text((bbox.yx[1] + 10, bbox.yx[0] + 10),
                        '%s\n%.2f' % (label, obj.score),
                        fill='red')

def main():
    
    synth = fluidsynth.Synth()

    synth.start('alsa')
    soundfont_id = synth.sfload('/usr/share/sounds/sf2/FluidR3_GM.sf2')
    for channel, instrument in enumerate(CHANNELS):
        synth.program_select(channel, soundfont_id, 0, instrument)
    
    pil_image = Image.open('PATH/TO/TEST/IMAGE/GOES/HERE.jpg') #this will open an image to run through PoseNet and return a Pose object

    pil_image.resize((641, 481), Image.NEAREST)
    engine = PoseEngine('models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
    poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
    print('Inference time: %.fms' % inference_time)

   # for pose in poses:

    velocities = {}
    for pose in poses:
        left = pose.keypoints.get('left wrist')
        right = pose.keypoints.get('right wrist')
        if not (left and right): continue

        if pose.score < 0.4: continue
        print('\nPose Score: ', pose.score)
        for label, keypoint in pose.keypoints.items():
            print(' %-20s x=%-4d y=%-4d score=%.1f' %
                  (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))
        
        identity = IDENTITIES[2 % len(IDENTITIES)] #set identity to VOICE_OOHS #Those Voice OOHs tho!
        left = 1 - left.yx[0] / engine.image_height
        right = 1 - right.yx[0] / engine.image_height
        velocity = int(left * 165)
        i = int(right * identity.extent)
        note = (identity.base_note
                + OCTAVE * (i // len(SCALE))
                + SCALE[i % len(SCALE)])
        velocities[(identity.channel, note)] = velocity
        # prev_notes is assumed to play notes based on the previous frame (when using video or a live feed
        #for note in prev_notes:
        #    if note not in velocities: synth.noteoff(*note)
        # Our way of making the note last longer since we are using just one frame
        for note, velocity in velocities.items():
             synth.noteon(*note, velocity)
             time.sleep(1)
        #for i, pose in enumerate(poses):
        #    identity = IDENTITIES[pose.id % len(IDENTITIES)]

if __name__ == '__main__':
    main()
