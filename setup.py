# -*- coding: utf-8 -*-

"""
Installation script
"""

from __future__ import absolute_import

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


install_requires = [
    "matplotlib>=2.0",
    "tensorflow-addons==0.13.0",
    "scipy",
    "numpy==1.19.2",
]

# If tensorflow-gpu is installed, use that
if get_dist("tensorflow") is None and get_dist("tensorflow-gpu") is not None:
    install_requires = [
        pkg.replace("tensorflow", "tensorflow-gpu") for pkg in install_requires
    ]

setup(
    name="hitman",
    description=(
        "approximate likelihood reconstruction for arbitrary neutrino detectors"
    ),
    author="Garrett Wendel et al.",
    author_email="gmw5164@psu.edu",
    url="tbd",
    license="Apache 2.0",
    version="0.01",
    python_requires=">=3.6",
    setup_requires=["pip>=1.8", "setuptools>18.5", "numpy==1.19.2", "flatbuffers==1.12.0", "gast==0.3.3", "grpcio==1.32.0", "six==1.15.0", "tensorflow-estimator==2.4.0", "typing-extensions==3.7.4", "wrapt==1.12.1", "google-auth==1.6.3"],
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    entry_points={
    'console_scripts': [
        'hitnet_train=hitman.train_hitnet:main','hitman_reco=hitman.hitman_reco:main','chargenet_train=hitman.train_chargenet:main'
    ],
},
)
