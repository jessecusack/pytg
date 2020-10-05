Finite difference based viscous Taylor-Goldstein equation solver.

This code is a python copy of [Bill Smyth's matlab code](http://blogs.oregonstate.edu/salty/matlab-tools-to-solve-the-viscous-taylor-goldstein-equation-for-both-instabilities-and-waves/) and all credit for should go to Bill, his colleagues and collaborators. If you use this code as part of a publication, you should cite:

[Smyth, W.D., J.N. Moum and J.D. Nash, 2011](https://oregonstate.app.box.com/s/jwkw46s0zv9oe93pudvtnpc5rg4kur9g): “Narrowband, high-frequency oscillations at the equator. Part II: Properties of shear instabilities”, *J. Phys. Oceanogr.* 41, 412-428.

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
```

If developing with jupyter notebook/lab it may be convenient to install the kernel as follows:
```
conda activate pytg
python -m ipykernel install --user --name pytg --display-name pytg
```