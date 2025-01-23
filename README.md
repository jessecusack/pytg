## Taylor-Goldstein equation solver

This code was developed from [Bill Smyth's matlab code](http://blogs.oregonstate.edu/salty/matlab-tools-to-solve-the-viscous-taylor-goldstein-equation-for-both-instabilities-and-waves/) and full credit for should go to Bill, his colleagues and collaborators. If you use this code as part of a publication, you should cite:

[Smyth, W.D., J.N. Moum and J.D. Nash, 2011](https://oregonstate.app.box.com/s/jwkw46s0zv9oe93pudvtnpc5rg4kur9g): “Narrowband, high-frequency oscillations at the equator. Part II: Properties of shear instabilities”, *J. Phys. Oceanogr.* 41, 412-428.

## Installation

Download the repository and install by running the following command in the base directory of the repository:

```bash
pip install .
```

## Development

Collaborative development is encouraged and anyone is welcome to submit issues or pull requests. 

To create a virtual environment using [uv](https://github.com/astral-sh/uv) that is accessible from VScode with the jupyter plugin:

```bash
cd pytg
uv sync
```