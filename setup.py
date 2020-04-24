from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
      name='slapp',
      use_scm_version=True,
      description=("Data pipeline and UI for human labeling of putative "
                   "ROIs from 2p cell segmentations"),
      author="Kat Schelonka, Isaak Willett, Dan Kapner, Nicholas Mei",
      author_email='kat.schelonka@alleninstitute.org',
      url="https://github.com/AllenInstitute/segmentation-labeling-app",
      packages=find_packages(),
      setup_requires=['setuptools_scm'],
      install_requires=required,
)
