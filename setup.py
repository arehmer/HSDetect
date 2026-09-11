from setuptools import setup, find_packages

with open("config/requirements.txt") as requirement_file:
    requirements = [line.rstrip() for line in requirement_file]




setup(
    name="HSDetect",
    description="A simple package.",
    version="0.0.1",
    author="Alexander Rehmer",
    author_email="rehmer@heimannsensor.com",
    install_requires=requirements,
    packages=find_packages(), # package = any folder with an __init__.py file
)