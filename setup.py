# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name='TFMTCNN',
    version='0.0.1',

    python_requires='>=3.5',

    packages=find_packages(),
    include_package_data=True,
    package_data={
        'tfmtcnn': [
            'data/*.txt',
            'images/*.jpg',
        ],
        'tfmtcnn.models': [
            'parameters/*/*'
        ]
    },

    install_requires=[
        'Click >= 7.0',
        'numpy >= 1.15.0',
        'h5py >= 2.9.0',
        'opencv-python >= 3.1.0.0, <= 3.4.5.20',
        'tensorflow >= 1.13.1',
        'tensorboard >= 1.13.1',
    ],

    extras_require={
        'GPU': ['tensorflow_gpu >= 1.13.1'],
    },

    entry_points={
        'console_scripts': [
            'example = tfmtcnn.apps.example:main',
            'lfw_metrics = tfmtcnn.apps.lfw_metrics:main',
            'lfw_pypi_metrics = tfmtcnn.apps.lfw_pypi_metrics:main',
        ],
    },

    url='https://github.com/RuslanKosarev/TFMTCNN',
    license='MIT',
    author='Ruslan Kosarev',
    author_email='ruslan.kosarev@gmail.com',

    description=(
        'MTCNN, a joint face detection and alignment using multi-task '
        'cascaded convolutional networks'
    ),
)
