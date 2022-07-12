from setuptools import setup

setup(name="EvoPlayground",
      version="0.1",
      description="A space to try out evolutionary techniques.",
      author="Anna Nickelson",
      author_email="nickelsa@oregonstate.edu",
      license="MIT",
      packages=["evoPlay"],
      install_requires=[
          "numpy"
      ],
      zip_safe=False)
