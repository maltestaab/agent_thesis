"""
setup.py - Setup script for the data science agents
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="data_science_agents",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.7",
) 