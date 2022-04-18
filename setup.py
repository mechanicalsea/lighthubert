import os
import subprocess
import sys

from setuptools import find_packages, setup


def write_version_py():
    with open(os.path.join("lighthubert", "version.txt")) as f:
        version = f.read().strip()

    # append latest commit hash to version string
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        version += "+" + sha[:7]
    except Exception:
        pass

    # write version info to lighthubert/version.py
    with open(os.path.join("lighthubert", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


version = write_version_py()


with open("README.md") as f:
    readme = f.read()


if "clean" in sys.argv[1:]:
    # Source: https://bit.ly/2NLVsgE
    print("deleting Cython files...")
    import subprocess

    subprocess.run(
        ["rm -f lighthubert/*.so lighthubert/**/*.so lighthubert/*.pyd lighthubert/**/*.pyd"],
        shell=True,
    )


requirements = [
    "torch>=1.8.1",
    "torchaudio>=0.8.1",
    "torchvision>=0.9.1",
    "numpy>=1.20.3",
]


def do_setup(package_data):
    setup(
        name="lighthubert",
        version=version,
        description="LightHuBERT: Lightweight and Configurable Speech Representation Learning with Once-for-All Hidden-Unit BERT",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/mechanicalsea/lighthubert",
        author="LightHuBERT authors",
        author_email="rwang@tongji.edu.cn",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence :: Speech",
        ],
        keywords="speech pre-training, model compression, knowledge distillation, neural architecture search, Transformer",
        
        packages=find_packages(
            include=[
                "lighthubert",
                "lighthubert.*"
            ],
            exclude=[
                "config",
                "config.*",
                "tutorials",
            ]
        ),
        # package_data=package_data,
        python_requires=">=3.8",
        install_requires=requirements,
    )


if __name__ == "__main__":
    do_setup(package_data=None)
