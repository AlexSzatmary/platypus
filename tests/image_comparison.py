# Taken from yhat's ggplot:
# https://github.com/yhat/ggplot

# Copyright (c) 2013, yhat
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib as mpl
import matplotlib.pyplot as plt
from nose.tools import with_setup, make_decorator, assert_true
import warnings


figsize_orig = mpl.rcParams["figure.figsize"]
def setup_package():
    mpl.rcParams["figure.figsize"] = (11.0, 8.0)


def teardown_package():
    mpl.rcParams["figure.figsize"] = figsize_orig



import os

# Testing framework shamelessly stolen from matplotlib...

# Tests which should be run with 'python tests.py' or via 'must be
# included here.
default_test_modules = [
    'ggplot.tests.test_basic',
    'ggplot.tests.test_readme_examples',
    'ggplot.tests.test_ggplot_internals',
    'ggplot.tests.test_geom',
    'ggplot.tests.test_stat',
    'ggplot.tests.test_stat_calculate_methods',
    'ggplot.tests.test_stat_summary',
    'ggplot.tests.test_geom_rect',
    'ggplot.tests.test_geom_dotplot',
    'ggplot.tests.test_geom_bar',
    'ggplot.tests.test_qplot',
    'ggplot.tests.test_geom_lines',
    'ggplot.tests.test_geom_linerange',
    'ggplot.tests.test_geom_pointrange',
    'ggplot.tests.test_faceting',
    'ggplot.tests.test_stat_function',
    'ggplot.tests.test_scale_facet_wrap',
    'ggplot.tests.test_scale_log',
    'ggplot.tests.test_reverse',
    'ggplot.tests.test_ggsave',
    'ggplot.tests.test_theme_mpl',
    'ggplot.tests.test_colors',
    'ggplot.tests.test_chart_components',
    'ggplot.tests.test_legend',
    'ggplot.tests.test_element_target',
    'ggplot.tests.test_element_text',
    'ggplot.tests.test_theme',
    'ggplot.tests.test_theme_bw',
    'ggplot.tests.test_theme_gray',
    'ggplot.tests.test_theme_mpl',
    'ggplot.tests.test_theme_seaborn'
]

_multiprocess_can_split_ = True


# Check that the test directories exist
if not os.path.exists(os.path.join(
        os.path.dirname(__file__), 'baseline_images')):
    raise IOError(
        'The baseline image directory does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install ggplot from source to get the '
        'test data.')

def _assert_same_ggplot_image(gg, name, test_file, tol=17):
    """Asserts that the ggplot object produces the right image"""
    fig = gg.draw()
    return _assert_same_figure_images(fig, name, test_file, tol=tol)

class ImagesComparisonFailure(Exception):
    pass

def _assert_same_figure_images(fig, name, test_file, tol=17):
    """Asserts that the figure object produces the right image"""
    import os
    import shutil
    from matplotlib import cbook
    from matplotlib.testing.compare import compare_images
    from nose.tools import assert_is_not_none

    if not ".png" in name:
        name = name+".png"

    basedir = os.path.abspath(os.path.dirname(test_file))
    basename = os.path.basename(test_file)
    subdir = os.path.splitext(basename)[0]

    baseline_dir = os.path.join(basedir, 'baseline_images', subdir)
    result_dir = os.path.abspath(os.path.join('result_images', subdir))

    if not os.path.exists(result_dir):
        cbook.mkdirs(result_dir)

    orig_expected_fname = os.path.join(baseline_dir, name)
    actual_fname = os.path.join(result_dir, name)

    def make_test_fn(fname, purpose):
        base, ext = os.path.splitext(fname)
        return '%s-%s%s' % (base, purpose, ext)
    expected_fname = make_test_fn(actual_fname, 'expected')
    # Save the figure before testing whether the original image
    # actually exists. This make creating new tests much easier,
    # as the result image can afterwards just be copied.
    fig.savefig(actual_fname)
    if os.path.exists(orig_expected_fname):
        shutil.copyfile(orig_expected_fname, expected_fname)
    else:
        raise Exception("Baseline image %s is missing" % orig_expected_fname)
    err = compare_images(expected_fname, actual_fname,
                         tol, in_decorator=True)
    if err:
        msg = 'images not close: {actual:s} vs. {expected:s} (RMS {rms:.2f})'.format(**err)
        raise ImagesComparisonFailure(msg)
    return err

def get_assert_same_ggplot(test_file):
    """Returns a "assert_same_ggplot" function for these test file
    call it like `assert_same_ggplot = get_assert_same_ggplot(__file__)`
    """
    def curried(*args, **kwargs):
        kwargs["test_file"] = test_file
        return _assert_same_ggplot_image(*args, **kwargs)
    curried.__doc__ = _assert_same_ggplot_image.__doc__
    return curried


def assert_same_elements(first,second, msg=None):
    assert_true(len(first) == len(second), "different length")
    assert_true(all([a==b for a,b in zip(first,second)]), "Unequal: %s vs %s" % (first, second))


def image_comparison(baseline_images=None, tol=17, extensions=None):
    """
    call signature::
      image_comparison(baseline_images=['my_figure'], tol=17)
    Compare images generated by the test with those specified in
    *baseline_images*, which must correspond else an
    ImagesComparisonFailure exception will be raised.
    Keyword arguments:
      *baseline_images*: list
        A list of strings specifying the names of the images generated
        by calls to :meth:`matplotlib.figure.savefig`.
      *tol*: (default 13)
        The RMS threshold above which the test is considered failed.
    """

    if baseline_images is None:
        raise ValueError('baseline_images must be specified')

    if extensions:
        # ignored, only for compatibility with matplotlibs decorator!
        pass

    def compare_images_decorator(func):
        import inspect
        _file = inspect.getfile(func)
        def decorated():
            # make sure we don't carry over bad images from former tests.
            assert len(plt.get_fignums()) == 0, "no of open figs: %s -> find the last test with ' " \
                                        "python tests.py -v' and add a '@cleanup' decorator." % \
                                        str(plt.get_fignums())
            func()
            assert len(plt.get_fignums()) == len(baseline_images), "different number of " \
                                                                   "baseline_images and actuall " \
                                                                   "plots."
            for fignum, baseline in zip(plt.get_fignums(), baseline_images):
                figure = plt.figure(fignum)
                _assert_same_figure_images(figure, baseline, _file, tol=tol)
        # also use the cleanup decorator to close any open figures!
        return make_decorator(cleanup(func))(decorated)
    return compare_images_decorator

def cleanup(func):
    """Decorator to add cleanup to the testing function
      @cleanup
      def test_something():
          " ... "
    Note that `@cleanup` is useful *only* for test functions, not for test
    methods or inside of TestCase subclasses.
    """

    def _teardown():
        plt.close('all')
        warnings.resetwarnings() #reset any warning filters set in tests

    return with_setup(setup=_setup, teardown=_teardown)(func)


# This is called from the cleanup decorator
def _setup():
    # The baseline images are created in this locale, so we should use
    # it during all of the tests.
    import locale
    import warnings
    from matplotlib.backends import backend_agg, backend_pdf, backend_svg

    try:
        locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, str('English_United States.1252'))
        except locale.Error:
            warnings.warn(
                "Could not set locale to English/United States. "
                "Some date-related tests may fail")

    # mpl.use('Agg', warn=False)  # use Agg backend for these tests
    # if mpl.get_backend().lower() != "agg":
    #     raise Exception(("Using a wrong matplotlib backend ({0}), which will not produce proper "
    #                     "images").format(mpl.get_backend()))

    # These settings *must* be hardcoded for running the comparison
    # tests
    mpl.rcdefaults()  # Start with all defaults
    mpl.rcParams['text.hinting'] = True
    mpl.rcParams['text.antialiased'] = True
    #mpl.rcParams['text.hinting_factor'] = 8

    # Clear the font caches.  Otherwise, the hinting mode can travel
    # from one test to another.
    backend_agg.RendererAgg._fontd.clear()
    backend_pdf.RendererPdf.truetype_font_cache.clear()
    backend_svg.RendererSVG.fontd.clear()
    # make sure we don't carry over bad plots from former tests
    assert len(plt.get_fignums()) == 0, "no of open figs: %s -> find the last test with ' " \
                                        "python tests.py -v' and add a '@cleanup' decorator." % \
                                        str(plt.get_fignums())


# This is here to run it like "from ggplot.tests import test; test()"
def test(verbosity=1):
    """run the ggplot test suite"""
    old_backend = mpl.rcParams['backend']
    try:
        mpl.use('agg')
        import nose
        import nose.plugins.builtin
        from matplotlib.testing.noseclasses import KnownFailure
        from nose.plugins.manager import PluginManager
        from nose.plugins import multiprocess

        # store the old values before overriding
        plugins = []
        plugins.append( KnownFailure() )
        plugins.extend( [plugin() for plugin in nose.plugins.builtin.plugins] )

        manager = PluginManager(plugins=plugins)
        config = nose.config.Config(verbosity=verbosity, plugins=manager)

        # Nose doesn't automatically instantiate all of the plugins in the
        # child processes, so we have to provide the multiprocess plugin with
        # a list.
        multiprocess._instantiate_plugins = [KnownFailure]

        success = nose.run( defaultTest=default_test_modules,
                            config=config,
                            )
    finally:
        if old_backend.lower() != 'agg':
            mpl.use(old_backend)

    return success

test.__test__ = False # nose: this function is not a test
