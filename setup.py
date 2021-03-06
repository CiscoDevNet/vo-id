import os.path
from setuptools import setup

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# This call to setup() does all the work
setup(
    name="void",
    version="0.1",
    description="Machine Learning Tools for Audio Processing",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/CiscoDevNet/vo-id",
    author="Real Python",
    author_email="dariocazzani@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8.6",
    ],
    packages=["void"],
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "void=void.__main__:main",
        ]
    },
)
