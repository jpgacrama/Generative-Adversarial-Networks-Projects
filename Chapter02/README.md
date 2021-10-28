## 3D-GAN - Generating Shapes Using GANs
## THERE IS A BUG IN TENSORFLOW FOR MACOS. This is working fine on a Windows laptop with CUDA support

Python 3.6

Steps to set up the project:
1. Create a python3 virtual environment and activate it
2. Install dependencies using "pip install -r requirements.txt"
3. Create essential folders like 1. logs 2. results 3. data
4. Download dataset to data directory
5. Train the model by executing "python3 run.py"

For Download:
1. wget http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip
2. unzip 3DShapeNetsCode.zip to data directory

Create the following directories:
1. models
2. logs
3. results