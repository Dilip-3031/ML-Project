from setuptools import find_packages, setup
from typing import List 


def get_requirements(file_path)->List[str]:
    '''
    this fumction will return the list of requirements
    '''
    requirements =[]

    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n',"")for req in requirements]
        
    return requirements








setup(
name = 'ML PROJECT KN',
version= '0.0.1',
author = 'dilip',
author_email='dparihar284003@gmail.com',
packages= find_packages(),
install_requires = get_requirements('requirements.txt')
)
