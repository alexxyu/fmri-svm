# fmri-svm
An SVM classifier trained to extract motion-specific information from fMRI scans of suppressed MT and V1 cortices.

## Setup

To set up a virtual environment and install dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Already-processed fMRI data for each subject can be found in the `scans` directory in the form of MAT files.

## Usage
* `svm_within.py` is used for within-subject testing.
* `svm_cp.py` and `svm_ip.py` are used for between-subject testing (CP-trained subjects and IP-trained subjects, respectively).
* `svm_combined.py` is used for between-subject testing, using both CP- and IP-trained subjects.

For specific usage, you can run a script with the `-h` or `--help` flag.
