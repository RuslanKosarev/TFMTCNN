# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import click
import cv2
import numpy as np

import tfmtcnn
from tfmtcnn.detector import FaceDetector
from tfmtcnn.prepare_data import ioutils

imgdir = tfmtcnn.dirname().joinpath('images')
outdir = tfmtcnn.dirname().joinpath(os.pardir, 'output')


@click.command()
@click.option('--detector', default='tfmtcnn', help='type of face detector, tfmtcnn or pypimtcnn')
@click.option('--show', default=True, help='show face detections')
def main(**args):
    detector = FaceDetector(detector=args['detector'])

    loader = ioutils.ImageLoaderWithPath(os.listdir(str(imgdir)), prefix=imgdir)

    for image, path in loader:
        boxes = detector.detect(image)
        print('\nimage', path, image.shape)
        print('number of detected faces', len(boxes))
        print(boxes)

        # save images
        ioutils.write_image(image, path.name, prefix=outdir)

        # show rectangles
        if args['show']:
            for bbox in boxes:
                position = (int(bbox[0]), int(bbox[1]))
                cv2.putText(image, str(np.round(bbox[4], 2)), position, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
                cv2.rectangle(image, position, (int(bbox[2]), int(bbox[3])), (0, 0, 255))

            cv2.imshow(str(path), image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
