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
    "matplotlib>=3.0",
    "scipy",
    "numpy>=1.19.2",
    "uproot>=3.14.4",
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
    version="0.2",
    python_requires=">=3.7",
    setup_requires=["pip>=1.8", "setuptools>30.5", "numpy>=1.19.2", "flatbuffers==2.0.7", "gast==0.4.0",
                    "grpcio>=1.47.1", "six==1.16.0", "tensorflow-estimator==2.10.0", "typing-extensions==4.4.0",
                    "wrapt==1.14.1", "google-auth==2.12.0"],
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'hitman_train=hitman.train_hitman:main', 'hitman_reco=hitman.hitman_reco:main'
        ],
    },
)
