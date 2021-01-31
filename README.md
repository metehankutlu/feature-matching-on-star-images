# Feature Matching on Star Images
Feature matching on star images using OpenCV, FAST feature detector, Brisk feature descriptor and Brute Force feature matching algorithm.

## Installation

```sh
git clone https://github.com/metehankutlu/feature-matching-on-star-images.git
```

### Using conda:

```sh
conda env create -f environment.yml
conda activate feature-matching
```

### Without conda:

```sh
pip install -r requirements.txt
```

## Usage

```sh
python main.py [-h] [--outMode {none,show,save}] [--outImage OUTIMAGE] [--drawMatches [DRAWMATCHES]] img1 img2

positional arguments:
  img1                  path for the first image
  img2                  path for the second image

optional arguments:
  -h, --help            show this help message and exit
  --outMode {none,show,save}
                        output mode
  --outImage OUTIMAGE   output image name
  --drawMatches [DRAWMATCHES]
                        show matched features
```