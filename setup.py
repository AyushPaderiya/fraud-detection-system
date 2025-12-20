from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements from file and return as list.
    Removes -e . if present (used for editable install).
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="fraud_detection",
    version="0.0.1",
    author="Ayush Paderiya",
    author_email="paderiyaayush@gmail.com",
    description="End-to-end fraud detection ML project",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
)
