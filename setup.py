from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirement(file_path: str) -> List[str]:
    """
    Reads a requirements.txt file and returns a list of dependencies.
    """
    requirement = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)

    return requirement

setup(
    name='mlproject',
    version='0.0.1',
    author='Ebenezer Reuben',
    author_email='ebenezerreuben447@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirement.txt')
)
