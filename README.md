
# SEQUENCE-TO-SUBSEQUENCE LEARNING WITH CONDITIONAL GAN FOR POWER DISAGGREGATION
This code implements the seqence-to-subseqence (seq2subseq) learning model. This model makes a trade-off between sequence-to-sequence (seq2seq) and sequence-to-point (seq2point) method, thus, balancing the convergence difficulty in deep neural networks and the amount of computation during the inference period. In addition, we apply U-Net [1] and Instance Normalization [2] techniques to our model. We build our model based on pix2pix [3] under Tensorflow framework [4]. Thanks a lot!


References:

[1] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical Image Computing &  Computer-assisted Intervention, 2015.

[2] Dmitry Ulyanov, Andrea Vedaldi, and Victor S. Lempitsky, “Instance normalization: The missing ingredient for fast stylization,” CoRR, vol. abs/1607.08022, 2016.

[3] https://phillipi.github.io/pix2pix/

[4] https://github.com/affinelayer/pix2pix-tensorflow

## Setup
- Create your own virtual environment with Python > 3.5
- Configure deep learning environment with Tensorflow (GPU edition) > 1.4.1 + cuDNN
- Install other necessary softwares, such as Matplotlib, Scikit-learn etc.
- Clone this repository

The environments we used are listed in the file `environment.yml`. If you use `conda`, you can use `conda env create -f environment.yml` to set up the environment.


## Datasets and preprocessing

### REFIT
Download the REFIT raw data from the original website (https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned). 


### UK-DALE
Download the UK-DALE raw data from the original website (https://ukerc.rl.ac.uk/DC/cgi-bin/edc_search.pl?GoButton=Detail&WantComp=41).


## Training & Test
We recommend importing the project into Pycharm (https://www.jetbrains.com/pycharm/) for your future research.

### Tips
You can look at the loss and computation graph using tensorboard:
```sh
tensorboard --logdir= 'the output path'
```


## Code validation and experiment results
We evaluate our model on a Linux machine with a Nvidia RTX 2080 GPU and Core i7 9700K CPU.

Example outputs of Washing Machine on ”unseen” house 2 from UK-DALE (We enlarge the right part of the figure to make it more clear):
![](/image/ukdale_washingmachine.png)


## Acknowledgments
This research is sponsored by National Key R&D Program of China(2017YFB0902600); State Grid Corporation of China Project (SGJS0000DKJS1700840) Research and Application of Key Technology for Intelligent Dispatching and Security Early-warning of Large Power Grid.
