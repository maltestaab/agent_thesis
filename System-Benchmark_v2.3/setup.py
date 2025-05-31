"""
setup.py - Setup script for the data science agents
"""

from setuptools import setup, find_packages

setup(
    name="data_science_agents",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "openai",
        "agents",
        "streamlit"
    ],
    python_requires=">=3.7",
) 