# fmralign
[Functional alignment and template estimation library](https://parietal-inria.github.io/fmralign-docs) for functional Magnetic Resonance Imaging (fMRI) data

This library is meant to be a light-weight Python library that handles functional
alignment tasks. It is compatible with and inspired by [Nilearn](http://nilearn.github.io).
Alternative implementations of these ideas can be found in the
[pymvpa](http://www.pymvpa.org) or [brainiak](http://brainiak.org) packages.

## Getting Started

### Installation

Open a terminal window, go the location where you want to install it. Then run:

```
pip install fmralign
```

Or if you want the latest version available (for example to develop):

```
git clone https://github.com/Parietal-INRIA/fmralign.git
cd fmralign
pip install -e .
```

Optionally, if you want to use optimal transport based method, you should also run:

```
pip install ott-jax
```

You're up and running!

### Documentation

You can found an introduction to functional alignment, a user guide and some examples
on how to use the package at https://parietal-inria.github.io/fmralign-docs.

## License

This project is licensed under the Simplified BSD License.

## Acknowledgments

This project has received funding from the European Union’s Horizon
2020 Research and Innovation Programme under Grant Agreement No. 785907
(HBP SGA2).
This project was supported by [Digiteo](http://www.digiteo.fr).
