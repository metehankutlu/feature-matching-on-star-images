import os
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import (read_imgs, get_keypoints, get_descriptors, 
                    get_matches, get_coordinates)

def main(img1path, img2path, drawMatches, outMode, outImage):

    img1, img2 = read_imgs(img1path, img2path)

    kp1, kp2 = get_keypoints(img1, img2)

    kp1, des1, kp2, des2 = get_descriptors(img1, kp1, img2, kp2)

    matches = get_matches(des1, des2)

    if len(matches) == 0:
        print('No match was found')
    else:
        coords = get_coordinates(img1, kp1, img2, kp2, matches)

        print('The coordinates of the rectangle found:')
        for i in coords:
            x, y = i[0]
            print('x:', x, 'y:', y)

        if outMode != 'none':
            output = cv2.polylines(img2, [np.int32(coords)], 
                                    True, 255, 3, cv2.LINE_AA)
            if drawMatches:
                output = cv2.drawMatchesKnn(
                    img1 = img1,
                    keypoints1 = kp1,
                    img2 = output,
                    keypoints2 = kp2,
                    matches1to2 = matches,
                    outImg = None,
                    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

            if outMode == 'show':
                plt.imshow(output)
                plt.axis('off')
                plt.show()
            elif outMode == 'save':
                outdir = 'output'
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                outImage = os.path.join(outdir, outImage)
                cv2.imwrite(outImage, output)

if __name__ == "__main__":
    args_config = [
        (
            'img1',
            {
                'help': 'path for the first image'
            }
        ),
        (
            'img2',
            {
                'help': 'path for the second image'
            }
        ),
        (
            '--outMode', 
            {
                'default': 'none', 
                'choices': ['none', 'show', 'save'], 
                'help': 'output mode'
            }
        ),
        (
            '--outImage', 
            {
                'default': 'output.jpg', 
                'help': 'output image name'
            }
        ),
        (
            '--drawMatches',
            {
                'help': 'show matched features',
                'nargs': '?',
                'default': False,
                'const': True,
                'type': bool
            }
        )
    ]
    parser = argparse.ArgumentParser()
    for name, kwargs in args_config:
        parser.add_argument(name, **kwargs)
    args = parser.parse_args()

    main(args.img1, args.img2, args.drawMatches, args.outMode, args.outImage)