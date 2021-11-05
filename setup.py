from setuptools import find_packages, setup

setup(
    name="gml-harm",
    packages=find_packages(include=["gml_harm"]),
    version="0.1.0",
    description="graphics media-lab image harmonization",
    author="Me",
    license="MIT",
    install_requires=[],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)