from setuptools import find_packages, setup

from surpbayes import __version__ as release

with open("requirements.txt", "r", encoding="utf-8") as f:
    # Make sure we strip all comments and options (e.g "--extra-index-url")
    # that arise from a modified pip.conf file that configure global options
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

setup(
    name="surpbayes",
    version=release,
    packages=find_packages(exclude=["tests"]),
    install_requires=requires,
    extras_require={
        "test": ["pytest>=6.1.2", "pytest-cov>=2.10.1"],
        "check": ["black>=20.8b1", "isort>=5.6.4", "mypy>=0.812", "pylint>=2.6.0"],
        "dev": [
            "isort>=5.6.4",
            "autoflake>=1.4",
            "ipython>=7.19.0",
            "notebook>=6.1.5",
            "jupyterlab>=2.2.9",
            "pip-tools>=5.4.0",
            "ipykernel>=5.3.4",
        ],
        "docs": [
            "sphinx==3.3.1",
            "sphinx_rtd_theme==0.5.0",
            "recommonmark==0.6.0",
            "sphinx-autodoc-typehints==1.11.1",
            "sphinx_copybutton==0.3.1",
            "nbsphinx==0.8.0",
            "docutils==0.17.1",
        ],
    },
)
