import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparse-learn",
    version="0.0.1",
    author="Baojian Zhou",
    author_email="bzhou6@albany.edu",
    description="A package related with sparse learning methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baojianzhou/sparse-learn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
