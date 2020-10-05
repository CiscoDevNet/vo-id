from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="dialogue",
    version="0.1",
    description="Machine Learning Tools for Audio Processing",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/dariocazzani/dialogue",
    author="Real Python",
    author_email="dariocazzani@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["reader"],
    include_package_data=True,
    install_requires=["torch", "torchaudio", "librosa"],
    entry_points={
        "console_scripts": [
            "dialogue=dialogue.__main__:main",
        ]
    },
)