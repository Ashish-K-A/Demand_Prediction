from setuptools import find_packages, setup
from typing import List
HYPHEN_E_DOT = '-e .'


def get_requirements(file_path:str)->List[str]:
	'''
	this functoin will return the list of requirements
	'''
	requirements=[]
	with open(file_path) as file_obj:
		requirements = file_obj.readlines() # when we use readlines() function \n is also added. Therefore we need to remove \n.
		requirements = [req.replace("\n","") for req in requirements]
		if HYPHEN_E_DOT in requirements:
			requirements.remove(HYPHEN_E_DOT)
	return requirements


setup(
    name='Demand_Prediction',
    version='0.0.1',
    author='Ashish',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)