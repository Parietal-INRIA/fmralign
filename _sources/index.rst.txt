fmralign
========

.. container:: index-paragraph

    fmralign is a Python module for **fast and easy functional alignment on fMRI** data.

    It leverages the `Nilearn <http://nilearn.github.io>`_ and
    `scikit-learn <http://scikit-learn.org>`_ toolboxes for alignment usecases,
    such as new data prediction or improved `decoding <https://nilearn.github.io/decoding/index.html>`_.


.. grid::

    .. grid-item-card:: :fas:`rocket` Quickstart
        :link: quickstart
        :link-type: ref
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Get started with fmralign

    .. grid-item-card:: :fas:`th` Examples
        :link: auto_examples/index.html
        :link-type: url
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Discover functionalities by reading examples

    .. grid-item-card:: :fas:`book` User guide
        :link: user_guide
        :link-type: ref
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Learn about functional alignment

Featured examples
-----------------

.. grid::

  .. grid-item-card::
    :link: auto_examples/plot_pairwise_alignment.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/images/sphx_glr_plot_pairwise_alignment_002.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Pairwise alignment

        Explore how to functionally align two subjects

  .. grid-item-card::
    :link: auto_examples/plot_template_alignment.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/images/sphx_glr_plot_template_alignment_002.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Template alignment

        Align a group of subjects to create a functional template

  .. grid-item-card::
    :link: auto_examples/plot_alignment_methods_benchmark.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/images/sphx_glr_plot_alignment_methods_benchmark_002.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Compare alignment methods

        Benchmark a range of alignment methods on a single dataset


.. toctree::
   :hidden:
   :includehidden:
   :titlesonly:

   quickstart.md
   auto_examples/index.rst
   user_guide.rst
   modules/index.rst

.. toctree::
   :hidden:
   :caption: Development

   authors.rst
   whats_new.rst
   GitHub Repository <https://github.com/parietal-INRIA/fmralign>


Nilearn is part of the :nipy:`NiPy ecosystem <>`.
