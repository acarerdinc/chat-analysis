from setuptools import setup, find_packages

setup(
    name="chat-analysis",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "chat-analysis=chat_analysis.cli:main",
        ],
    },
    install_requires=[
        'transformers',
        'tensorflow',
        'numpy',
        'PyYAML',
    ],
)
