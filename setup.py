# https://github.com/kennethreitz/setup.py

# standard libraries
import io
import os

# external libraries
from setuptools import find_packages, setup

# Package meta-data.
NAME = "gestrol"
DESCRIPTION = "gesture-based interface"
URL = "https://github.com/samvoisin/gesture-control"
EMAIL = "samvoisin@protonmail.com"
AUTHOR = "Sam Voisin"
REQUIRES_PYTHON = ">=3.8"
VERSION = "0.1.0"

# What packages are required for this module to be executed?
REQUIRED = []

# What packages are optional?
EXTRAS = {}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

# NOT CONCERNED ABOUT THIS FOR NOW
# class UploadCommand(Command):
#    """Support setup.py upload."""
#
#    description = "Build and publish the package."
#    user_options = []
#
#    @staticmethod
#    def status(s):
#        """Prints things in bold."""
#        print("\033[1m{0}\033[0m".format(s))
#
#    def initialize_options(self):
#        pass
#
#    def finalize_options(self):
#        pass
#
#    def run(self):
#        try:
#            self.status("Removing previous builds…")
#            rmtree(os.path.join(here, "dist"))
#        except OSError:
#            pass
#
#        self.status("Building Source and Wheel (universal) distribution…")
#        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))
#
#        self.status("Uploading the package to PyPI via Twine…")
#        os.system("twine upload dist/*")
#
#        self.status("Pushing git tags…")
#        os.system("git tag v{0}".format(about["__version__"]))
#        os.system("git push --tags")
#
#        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["test"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    # extras_require=EXTRAS,
    include_package_data=True,
    license="GNU"
    # $ setup.py publish support.
    # cmdclass={
    #    "upload": UploadCommand,
    # },
)
