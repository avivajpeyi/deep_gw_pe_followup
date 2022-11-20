Deep Follow-up of GW events in q-xeff space
===========================================

Software to compute the marginalised likelihood $Z(d| q, xeff)$ for a data $d$
containing a CBC gravitational wave event and a q-xeff point in parameter space.

The method is described in the paper
''`Deep follow-up of GW151226: ordinary binary or low-mass-ratio system? <https://arxiv.org/abs/2203.13406>`_''.


Studies
-------

The configurations for tests and examples of the software are available in the `studies` directory.

1. GW151226 like-injection

2. GW150914

3. GW151226

All results are uploaded in `Zenodo`_.

.. _Zenodo: https://zenodo.org/record/6975894
.. _Deep follow-up of GW151226: ordinary binary or low-mass-ratio system?


Installation and usage
----------------------

To install the software, clone this repository and run the following command:
`pip install -e .`

To generate a q-xeff prior and begin analysis, refer to the `studies/fast_injection` directory.
