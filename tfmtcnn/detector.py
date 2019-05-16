# coding:utf-8
__author__ = 'Ruslan N. Kosarev'


class MTCNN:
    def __init__(self):
        from mtcnn.mtcnn import MTCNN
        self.__detector = MTCNN().detect_faces

    def detector(self, image):
        faces = self.__detector(image)
        boxes = []

        for face in faces:
            box = face['box']
            box[2] += box[0]
            box[3] += box[1]
            boxes.append(box + [face['confidence']])

        return boxes


class TFMTCNN:
    def __init__(self):
        from tfmtcnn.mtcnn import MTCNN
        self.__detector = MTCNN().detect

    def detector(self, image):
        boxes, _ = self.__detector(image)
        return boxes


class FaceDetector:
    def __init__(self, detector='tfmtcnn'):

        if detector is 'pypimtcnn':
            self.__detector = MTCNN().detector

        elif detector is 'tfmtcnn':
            self.__detector = TFMTCNN().detector

        else:
            raise 'Undefined face detector type {}'.format(detector)

    def detect(self, image):
        return self.__detector(image)
