from setuptools import setup, find_packages

setup(
    name="package",
    version="0.1.0",
    packages=find_packages(include=["modules", "modules.*"]),
    description="A package for OCR, XAI, data augmentation, and sentient analyzer",
    author="Abhay-Cerberus, Priyanshi, Madhav, Ayushi",
    url="https://github.com/Abhay-Cerberus/Package",  
    install_requires=[
        "textblob"
    ],
)