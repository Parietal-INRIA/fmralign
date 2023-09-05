# fmralign

[Functional alignment for fMRI](https://parietal-inria.github.io/fmralign-docs) (functional Magnetic Resonance Imaging) data.

This light-weight Python library provides access to a range of functional alignment methods, including Procrustes and Optimal Transport.
It is compatible with and inspired by [Nilearn](http://nilearn.github.io).
Alternative implementations of these ideas can be found in the [pymvpa](http://www.pymvpa.org) or [brainiak](http://brainiak.org) packages.
The [netrep](https://github.com/ahwillia/netrep) library also offers many of the same metrics,though with a more general focus beyond fMRI data.

## Getting Started

### Installation

You can access the latest stable version of fmralign directly with the PyPi package installer:

```
pip install fmralign
```

For development or bleeding-edge features, fmralign can also be installed directly from source:

```
git clone https://github.com/Parietal-INRIA/fmralign.git
cd fmralign
pip install -e .
```

Note that if you want to use the JAX-accelerated optimal transport methods, you should also run:

```
pip install fmralign .[jax]
```

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
