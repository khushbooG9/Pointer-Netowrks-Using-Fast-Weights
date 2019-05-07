Setup and Code Dependencies –
1.	Either a gmail account to access https://colab.research.google.com  or Jupyter Notebook at the local machine.
2.  Python 3.7
3.	Tensorflow 1.13.1
4.	Matplotlib 3.0.3
5.	Numpy 1.14.6
6.	Random 

Note - boundary_PtrLSTM.ipynb has the boundary detection task implementation using LSTMcell in Pointer Network. PtrFW_boundary.ipynb has the boundary detection task implementation using Fast Weights module. Albation_Study_PtrFW_boundary.ipynb  is being used to perform the experiments using 5 different hyperparameters and ablation study.


How to run - 
1.  (Preferred) Each of the ipynb files has a link to run the code on google colab, which runs the code in playground mode(without affecting the original code). Make sure to go to the colab notebook settings and switch Runtime type to Python 3 and Hardware accelerator to GPU. All dependencies are preloaded here. 

2.  If unable to access the link in 1. , you can download the files on local machine and then upload them back to the google colab for similar setup.

3.  If running on local machine using Jupyter Notebook and not using GPU, please remove the code "with tf.device("/gpu:0"):" from the training cell and adjust the code formatting accordingly. Options 1 and 2 are recommended for hasslefree experience. 


**Other details such as various parameters or how to use the dataset generator functions are mentioned in between the code itself.
Please follow that information, for better clarity.**