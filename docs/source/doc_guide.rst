*Surreal* Code Documentation Guide
==================================

Documentation for functions and classes should follow the rules listed
below. They are taken from `Google Style
Guide <https://google.github.io/styleguide/pyguide.html>`__ and slightly
modified.

Sphinx guide
------------

Please refer to
`Sphinx-theme <https://github.com/LinxiFan/Sphinx-theme>`__ for more
details.

Demos of the new Sphinx themes can be viewed here:

-  `Stanford custom theme <https://linxifan.github.io/Sphinx-demo/>`__
   based on Stanford web color palette.
-  `Neo RTD theme <https://linxifan.github.io/Neo-RTD-theme-demo/>`__.

The doc webpages are generated with an enhanced `napolean
extension <https://github.com/LinxiFan/sphinxcontrib-napolean>`__.

Run the included ``./sphinx.sh`` script:

.. code:: bash

    usage: ./sphinx.sh <command> <args...>
    start:      automate "sphinx-quickstart" command
        args: <project_name> "<authors>" <version>
    extra:      append extra config code to the end of "docs/source/conf.py"
    p|py:       regenerate pydocs
        args: <package_name>
    b|build:    clean and rebuild html
    c|commit:   commit generated html to "gh-pages" branch
    push:       push "gh-pages" branch to remote

Doc Standards
-------------

Doc Strings
~~~~~~~~~~~

Python has a unique commenting style using doc strings. A doc string is
a string that is the first statement in a package, module, class or
function. These strings can be extracted automatically through the
``__doc__`` member of the object and are used by pydoc. (Try running
pydoc on your module to see how it looks.)

We always use the three double-quote ``"""`` format for doc strings (per
PEP 257). A doc string should be organized as a summary line (one
physical line) terminated by a period, question mark, or exclamation
point, followed by a blank line, followed by the rest of the doc string
starting at the same cursor position as the first quote of the first
line.

Functions and Methods
~~~~~~~~~~~~~~~~~~~~~

As used in this section "function" applies to methods, function, and
generators.

A function must have a docstring, unless it meets all of the following
criteria:

-  not externally visible
-  very short
-  obvious

A docstring should give enough information to write a call to the
function without reading the function's code. A docstring should
describe the function's calling syntax and its semantics, not its
implementation. For tricky code, comments alongside the code are more
appropriate than using docstrings.

Certain aspects of a function should be documented in special sections,
listed below. Each section begins with a heading line, which ends with a
colon. Sections should be indented two spaces, except for the heading.

``Args``
^^^^^^^^

List each parameter by name. A description should follow the name, and
be separated by a colon and a space. If the description is too long to
fit on a single 80-character line, use a hanging indent of **2 spaces**.

The description should mention required type(s) and the meaning of the
argument.

If a function accepts ``*foo`` (variable length argument lists) and/or
``**bar`` (arbitrary keyword arguments), they should be listed as
``*foo`` and ``**bar``.

``Returns`` (or ``Yields`` for generators)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Describe the type and semantics of the return value. If the function
only returns None, this section is not required.

``Raises``
^^^^^^^^^^

List all exceptions that are relevant to the interface.

Example
^^^^^^^

.. code:: python

    def fetch_bigtable_rows(big_table, keys, other_silly_variable=None):
        """Fetches rows from a Bigtable.
        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
          big_table: An open Bigtable Table instance.
          keys: A sequence of strings representing the key of each table row
            to fetch.
          other_silly_variable: Another optional variable, that has a much
            longer name than the other args, and which does nothing.

        Returns:
          A dict mapping keys to the corresponding table row data
          fetched. Each row is represented as a tuple of strings. For
          example:

          {'Serak': ('Rigel VII', 'Preparer'),
           'Zim': ('Irk', 'Invader'),
           'Lrrr': ('Omicron Persei 8', 'Emperor')}

          If a key from the keys argument is missing from the dictionary,
          then that row was not found in the table.

        Raises:
          IOError: An error occurred accessing the bigtable.Table object.
        """
        pass

Classes
~~~~~~~

Classes should have a doc string below the class definition describing
the class. If your class has public attributes, they should be
documented here in an Attributes section and follow the same formatting
as a function's ``Args`` section.

.. code:: python

    class SampleClass(object):
        """Summary of class here.

        Longer class information....
        Longer class information....

        Attributes:
            likes_spam: A boolean indicating if we like SPAM or not.
            eggs: An integer count of the eggs we have laid.
        """

        def __init__(self, likes_spam=False):
            """Inits SampleClass with blah."""
            self.likes_spam = likes_spam
            self.eggs = 0

        def public_method(self):
            """Performs operation blah."""

Block and Inline Comments
~~~~~~~~~~~~~~~~~~~~~~~~~

The final place to have comments is in tricky parts of the code. If
you're going to have to explain it at the next code review, you should
comment it now. Complicated operations get a few lines of comments
before the operations commence. Non-obvious ones get comments at the end
of the line.

.. code:: python

    # We use a weighted dictionary search to find out where i is in
    # the array.  We extrapolate position based on the largest num
    # in the array and the array size and then do binary search to
    # get the exact number.

    if i & (i-1) == 0:        # true iff i is a power of 2

To improve legibility, these comments should be at least 2 spaces away
from the code.

On the other hand, never describe the code. Assume the person reading
the code knows Python (though not what you're trying to do) better than
you do.
