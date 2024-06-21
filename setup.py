import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as reqs:
    requirements = reqs.read()

setuptools.setup(
    name="impact2_engine",
    version="0.0.1",
    author="Clinstat",
    description="Impact Python package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/clinstatdevice-main/impact2_monitoring.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=[requirements]
)