#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools


setuptools.setup(
    name='strain-detangler',
    version='0.1.0',
    description="",
    author="Lauren Mak",
    author_email='',
    url='',
    packages=setuptools.find_packages(),
    package_dir={'strain-detangler': 'strain-detangler'},
    install_requires=[
        'click',
        'pandas',
        'scipy',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'strain-detangler=strain_detangler.cli:main'
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
