# :book: HTML documentation

QUEENS uses [Sphinx](https://www.sphinx-doc.org/en/master/) to automatically build an html-documentation from the docstrings.

## :woman_teacher: Build the documentation

To build the documentation, you first need a development QUEENS installation as mentioned in the [CONTRIBUTING.md](../CONTRIBUTING.md). This adds additional packages needed for the html build. Once this is done, follow the next steps in the activated Python environment with the development installation.

When building the documentation on your machine for the first time or after adding new modules or classes to QUEENS, one needs to first rebuild the `autodoc index` by running:

```bash
cd <queens-base-directory>
sphinx-apidoc -o doc/source queens -f -M
```

To actually build the html-documentation, navigate into the doc folder and run the make command:

```bash
cd doc
sphinx-build -b html -d build/doctrees source build/html
```

You can now view the documentation in your favorite browser by opening `build/html/index.html`.
