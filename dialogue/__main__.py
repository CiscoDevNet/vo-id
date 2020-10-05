"""Machine Learning tools for audio processing
Usage:
------
    $ dialogue <audio-path>
    -h, --help         Show this help

Contact:
--------
- dariocazzani@gmail.com
- https://github.com/dariocazzani/

Version:
--------
- dialogue v0.1
"""

import argparse
parser = argparse.ArgumentParser(description='', add_help=False)
parser.add_argument('-v', '--version', action='version',
                    version='dialogue version=0.1')
parser.add_argument('-h', '--help', action='store_true')

args = parser.parse_args()

def main():  # type: () -> None
    if args.help:
        print(__doc__)

if __name__ == "__main__":
    main()