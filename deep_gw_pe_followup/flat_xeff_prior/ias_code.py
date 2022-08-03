"""Utility functions."""

import importlib
import inspect
import json
import os
import pathlib
import re
import sys
import tempfile
import textwrap
import numpy as np
import itertools
from abc import ABC, abstractmethod

DIR_PERMISSIONS = 0o755
FILE_PERMISSIONS = 0o644


class ClassProperty:
    """
    Can be used like `@property` but for class attributes instead of
    instance attributes.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, inst, cls):
        return self.func(cls)


def merge_dictionaries_safely(dics):
    """
    Merge multiple dictionaries into one.
    Accept repeated keys if values are consistent, otherwise raise
    `ValueError`.
    """
    merged = {}
    for dic in dics:
        for key in merged.keys() & dic.keys():
            if merged[key] != dic[key]:
                raise ValueError(f'Found incompatible values for {key}')
        merged |= dic
    return merged


def mkdirs(dirname, dir_permissions=DIR_PERMISSIONS):
    """
    Create directory and its parents if needed, ensuring the
    whole tree has the same permissions. Existing directories
    are left unchanged.

    Parameters
    ----------
    dirname: path of directory to make.
    dir_permissions: octal with permissions.
    """
    dirname = pathlib.Path(dirname)
    for path in list(dirname.parents)[::-1] + [dirname]:
        path.mkdir(mode=dir_permissions, exist_ok=True)


# ----------------------------------------------------------------------
# JSON I/O:

class_registry = {}



class JSONMixin:
    """
    Provide JSON output to subclasses.
    Register subclasses in `class_registry`.

    Define a method `get_init_dict` which works for classes that store
    their init parameters as attributes with the same names. If this is
    not the case, the subclass should override `get_init_dict`.

    Define a method `reinstantiate` that allows to safely modify
    attributes defined at init.
    """

    def to_json(self, dirname, basename=None, *,
                dir_permissions=DIR_PERMISSIONS,
                file_permissions=FILE_PERMISSIONS, overwrite=False):
        """
        Write class instance to json file.
        It can then be loaded with `read_json`.
        """
        basename = basename or f'{self.__class__.__name__}.json'
        filepath = pathlib.Path(dirname) / basename

        if not overwrite and filepath.exists():
            raise FileExistsError(
                f'{filepath.name} exists. Pass `overwrite=True` to overwrite.')

        mkdirs(dirname, dir_permissions)

        with open(filepath, 'w') as outfile:
            json.dump(self, outfile, cls=CogwheelEncoder, dirname=dirname,
                      file_permissions=file_permissions, overwrite=overwrite,
                      indent=2)
        filepath.chmod(file_permissions)

    def __init_subclass__(cls):
        """Register subclasses."""
        super().__init_subclass__()
        class_registry[cls.__name__] = cls

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to `__init__`.
        Only works if the class stores its init parameters as attributes
        with the same names. Otherwise, the subclass should override
        this method.
        """
        keys = list(inspect.signature(self.__init__).parameters)
        if any(not hasattr(self, key) for key in keys):
            raise KeyError(
                f'`{self.__class__.__name__}` must override `get_init_dict` '
                '(or store its init parameters with the same names).')
        return {key: getattr(self, key) for key in keys}

    def reinstantiate(self, **new_init_kwargs):
        """
        Return an new instance of the current instance's class, with an
        option to update `init_kwargs`. Values not passed will be taken
        from the current instance.
        """
        init_kwargs = self.get_init_dict()

        if not new_init_kwargs.keys() <= init_kwargs.keys():
            raise ValueError(
                f'`new_init_kwargs` must be from ({", ".join(init_kwargs)})')

        return self.__class__(**init_kwargs | new_init_kwargs)


class Prior(ABC, JSONMixin):
    """"
    Abstract base class to define priors for Bayesian parameter
    estimation, together with coordinate transformations from "sampled"
    parameters to "standard" parameters.

    Schematically,
        lnprior(*sampled_par_vals, *conditioned_on_vals)
            = log P(sampled_par_vals | conditioned_on_vals)
    where P is the prior probability density in the space of sampled
    parameters;
        transform(*sampled_par_vals, *conditioned_on_vals)
            = standard_par_dic
    and
        inverse_transform(*standard_par_vals, *conditioned_on_vals)
            = sampled_par_dic.

    Subclassed by `CombinedPrior` and `FixedPrior`.

    Attributes
    ----------
    range_dic: Dictionary whose keys are sampled parameter names and
               whose values are pairs of floats defining their ranges.
               Needs to be defined by the subclass (either as a class
               attribute or instance attribute) before calling
               `Prior.__init__()`.
    sampled_params: List of sampled parameter names (keys of range_dic)
    standard_params: List of standard parameter names.
    conditioned_on: List of names of parameters on which this prior
                    is conditioned on. To combine priors, conditioned-on
                    parameters need to be among the standard parameters
                    of another prior.
    periodic_params: List of names of sampled parameters that are
                     periodic.

    Methods
    -------
    lnprior: Method that takes sampled and conditioned-on parameters
             and returns a float with the natural logarithm of the prior
             probability density in the space of sampled parameters.
             Provided by the subclass.
    transform: Coordinate transformation, function that takes sampled
               parameters and conditioned-on parameters and returns a
               dict of standard parameters. Provided by the subclass.
    lnprior_and_transform: Take sampled parameters and return a tuple
                           with the result of (lnprior, transform).
    inverse_transform: Inverse coordinate transformation, function that
                       takes standard parameters and conditioned-on
                       parameters and returns a dict of sampled
                       parameters. Provided by the subclass.
    """

    conditioned_on = []
    periodic_params = []

    def __init__(self, **kwargs):
        super().__init__()
        self._check_range_dic()

        self.cubemin = np.array([rng[0] for rng in self.range_dic.values()])
        cubemax = np.array([rng[1] for rng in self.range_dic.values()])
        self.cubesize = cubemax - self.cubemin
        self.signature = inspect.signature(self.transform)

    @ClassProperty
    def sampled_params(self):
        """List of sampled parameter names."""
        return list(self.range_dic)

    @ClassProperty
    @abstractmethod
    def range_dic(self):
        """
        Dictionary whose keys are sampled parameter names and
        whose values are pairs of floats defining their ranges.
        Needs to be defined by the subclass.
        If the ranges are not known before class instantiation,
        define a class attribute as {'<par_name>': NotImplemented, ...}
        and populate the values at the subclass' `__init__()` before
        calling `Prior.__init__()`.
        """
        return {}

    @ClassProperty
    @abstractmethod
    def standard_params(self):
        """
        List of standard parameter names.
        """
        return []

    @abstractmethod
    def lnprior(self, *par_vals, **par_dic):
        """
        Natural logarithm of the prior probability density.
        Take `self.sampled_params + self.conditioned_on` parameters and
        return a float.
        """

    @abstractmethod
    def transform(self, *par_vals, **par_dic):
        """
        Transform sampled parameter values to standard parameter values.
        Take `self.sampled_params + self.conditioned_on` parameters and
        return a dictionary with `self.standard_params` parameters.
        """

    @abstractmethod
    def inverse_transform(self, *par_vals, **par_dic):
        """
        Transform standard parameter values to sampled parameter values.
        Take `self.standard_params + self.conditioned_on` parameters and
        return a dictionary with `self.sampled_params` parameters.
        """

    def lnprior_and_transform(self, *par_vals, **par_dic):
        """
        Return a tuple with the results of `self.lnprior()` and
        `self.transform()`.
        The reason for this function is that for `CombinedPrior` it is
        already necessary to evaluate `self.transform()` in order to
        evaluate `self.lnprior()`. `CombinedPrior` overwrites this
        function so the user can get both `lnprior` and `transform`
        without evaluating `transform` twice.
        """
        return (self.lnprior(*par_vals, **par_dic),
                self.transform(*par_vals, **par_dic))

    def _check_range_dic(self):
        """
        Ensure that range_dic values are stored as float arrays.
        Verify that ranges for all periodic parameters were
        provided.
        """
        if missing := (set(self.periodic_params) - self.range_dic.keys()):
            raise PriorError('Periodic parameters are missing from '
                             f'`range_dic`: {", ".join(missing)}')

        for key, value in self.range_dic.items():
            if not hasattr(value, '__len__') or len(value) != 2:
                raise PriorError(f'`range_dic` {self.range_dic} must have '
                                 'ranges defined as pair of floats.')
            self.range_dic[key] = np.asarray(value, dtype=np.float_)

    def __init_subclass__(cls):
        """
        Check that subclasses that change the `__init__` signature also
        define their own `get_init_dict` method.
        """
        super().__init_subclass__()

        if (inspect.signature(cls.__init__)
                != inspect.signature(Prior.__init__)
                and cls.get_init_dict is Prior.get_init_dict):
            raise PriorError(
                f'{cls.__name__} must override `get_init_dict` method.')

    def __repr__(self):
        """
        Return a string of the form
        `Prior(sampled_params | conditioned_on) → standard_params`.
        """
        rep = self.__class__.__name__ + f'({", ".join(self.sampled_params)}'
        if self.conditioned_on:
            rep += f' | {", ".join(self.conditioned_on)}'
        rep += f') → [{", ".join(self.standard_params)}]'
        return rep

    @staticmethod
    def get_init_dict():
        """
        Return dictionary with keyword arguments to reproduce the class
        instance. Subclasses should override this method if they require
        initialization parameters.
        """
        return {}


class UniformPriorMixin:
    """
    Define `lnprior` for uniform priors.
    It must be inherited before `Prior` (otherwise a `PriorError` is
    raised) so that abstract methods get overriden.
    """

    def lnprior(self, *par_vals, **par_dic):
        """
        Natural logarithm of the prior probability density.
        Take `self.sampled_params + self.conditioned_on` parameters and
        return a float.
        """
        return - np.log(np.prod(self.cubesize))

    def __init_subclass__(cls):
        """
        Check that UniformPriorMixin comes before Prior in the MRO.
        """
        super().__init_subclass__()
        check_inheritance_order(cls, UniformPriorMixin, Prior)


class PriorError(Exception):
    """Base class for all exceptions in this module"""


def check_inheritance_order(subclass, base1, base2):
    """
    Check that class `subclass` subclasses `base1` and `base2`, in that
    order. If it doesn't, raise `PriorError`.
    """
    for base in base1, base2:
        if not issubclass(subclass, base):
            raise PriorError(
                f'{subclass.__name__} must subclass {base.__name__}')

    if subclass.mro().index(base1) > subclass.mro().index(base2):
        raise PriorError(f'Wrong inheritance order: `{subclass.__name__}` '
                         f'must inherit from `{base1.__name__}` before '
                         f'`{base2.__name__}` (or their subclasses).')


class FlatChieffPrior(UniformPriorMixin, Prior):
    """
    Spin prior for aligned spins that is flat in effective spin chieff.
The sampled parameters are `chieff` and `cumchidiff`.
`cumchidiff` ranges from 0 to 1 and is typically poorly measured.
`cumchidiff` is the cumulative of a prior on the spin difference
that is uniform conditioned on `chieff`, `q`.
    """
    standard_params = ['s1z', 's2z']
    range_dic = {'chieff': (-1, 1),
                 'cumchidiff': (0, 1)}
    conditioned_on = ['m1', 'm2']

    @staticmethod
    def _get_s1z(chieff, q, s2z):
        return (1 + q) * chieff - q * s2z

    def _s1z_lim(self, chieff, q):
        s1z_min = np.maximum(self._get_s1z(chieff, q, s2z=1), -1)
        s1z_max = np.minimum(self._get_s1z(chieff, q, s2z=-1), 1)
        return s1z_min, s1z_max

    def transform(self, chieff, cumchidiff, m1, m2):
        """(chieff, cumchidiff) to (s1z, s2z)."""
        q = m2 / m1
        s1z_min, s1z_max = self._s1z_lim(chieff, q)
        s1z = s1z_min + cumchidiff * (s1z_max - s1z_min)
        s2z = ((1 + q) * chieff - s1z) / q
        return {'s1z': s1z,
                's2z': s2z}

    def inverse_transform(self, s1z, s2z, m1, m2):
        """(s1z, s2z) to (chieff, cumchidiff)."""
        q = m2 / m1
        chieff = (s1z + q * s2z) / (1 + q)
        s1z_min, s1z_max = self._s1z_lim(chieff, q)
        cumchidiff = (s1z - s1z_min) / (s1z_max - s1z_min)
        return {'chieff': chieff,
                'cumchidiff': cumchidiff}

    def add_spin_samples(self, dictlike_samples):
        """
        Given dict-like object of samples including `m1` and `m2`,
        add to the object uniform samples of `chieff` and `cumchidiff`
        along with the corresponding values of `s1z` and `s2z`
        """
        nsamples = 1
        if hasattr(dictlike_samples['m1'], '__len__'):
            nsamples = len(dictlike_samples['m1'])
        dictlike_samples['chieff'] = np.random.uniform(
            *self.range_dic['chieff'], nsamples)
        dictlike_samples['cumchidiff'] = np.random.uniform(
            *self.range_dic['cumchidiff'], nsamples)
        zspindic = self.transform(dictlike_samples['chieff'],
                                  dictlike_samples['cumchidiff'],
                                  dictlike_samples['m1'], dictlike_samples['m2'])
        for k in ['s1z', 's2z']:
            dictlike_samples[k] = zspindic[k]


####
## ADDED BY AVI BELOW

def get_masses_samples(N):
    mass_ratio = np.random.uniform(0.125, 1, N)
    chirp_mass = np.random.uniform(25, 31, N)
    with np.errstate(invalid="ignore"):
        total_mass = chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio ** 0.6
        mass_1 = total_mass / (1 + mass_ratio)
        mass_2 = mass_1 * mass_ratio
        return mass_1, mass_2


def test_ias_prior():
    from corner import corner
    import pandas as pd
    masses = get_masses_samples(N=10000)
    samples = {'m1': masses[0], 'm2': masses[1]}
    FlatChieffPrior().add_spin_samples(samples)
    df = pd.DataFrame(samples)
    fig = corner(df, labels=df.keys(),
                 bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                 title_kwargs=dict(fontsize=16), color='#0072C1',
                 truth_color='tab:orange', quantiles=[0.16, 0.84],
                 levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                 plot_density=False, plot_datapoints=True, fill_contours=True,
                 max_n_ticks=3, hist_kwargs=dict(density=True))
    fig.savefig("ias_test.png")


if __name__ == '__main__':
    test_ias_prior()
