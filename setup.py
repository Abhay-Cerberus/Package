from setuptools import setup, find_packages

setup(
    name="package",  # change to your package's name
    version="0.1.0",
    packages=find_packages(include=["modules", "modules.*"]),
    description="A package for OCR, XAI, data augmentation, and edge optimization",
    author="Abhay-Cerberus, Priyanshi, Madhav, Ayushi",
    url="https://github.com/Abhay-Cerberus/Package",  # optional
    install_requires=[
        "tensorflow>=2.19.0",
        "torch",
        "tensorflow-model-optimization>=0.8.0"
    ],
)
