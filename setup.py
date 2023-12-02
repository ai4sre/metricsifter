from setuptools import find_namespace_packages, setup

with open("README.md", "r", encoding="UTF-8") as readme:
    long_description = readme.read()
with open("requirements.txt", "r", encoding="UTF-8") as requirements:
    install_requires = requirements.read().split("\n")


setup(
    name="metricsifter",
    version="0.0.1",
    author="yuuki",
    description="Feature Reduction for Multivariate Time Series Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai4sre/metricsifter",
    project_urls={
        "Bug Tracker": "https://github.com/ai4sre/metricsifter/issues",
    },
    packages=find_namespace_packages(include="metricsifter.*"),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    license="MIT",
)
