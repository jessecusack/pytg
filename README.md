Finite difference based viscous Taylor-Goldstein equation solver.

This code is python copy of [Bill Smyth's matlab code](http://blogs.oregonstate.edu/salty/matlab-tools-to-solve-the-viscous-taylor-goldstein-equation-for-both-instabilities-and-waves/) and all credit for should go to Bill, his colleagues and collaborators. 

Download the repository and install by running the following command in the base directory of the repository:
```
pip install -e .
```

Collaborative development is encouraged and anyone is welcome to submit issues or pull requests. 

For development, install the package in a new conda environment as follows:
```
conda env create -f environment.yml
conda activate pytg
pip install -e .
python -m ipykernel install --user --name pytg --display-name pytg
```

