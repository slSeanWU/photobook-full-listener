import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="photobook-full-listener",
    version="1.0.0",
    author="Lorenzo_Santos",
    author_email="lorenzodavinci@hotmail.com",
    description="This is a repo for listener model for the PhotoBook Referential Game with CLIPScores as Implicit Reference Chain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lorenzo7377/photobook-full-listener",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)