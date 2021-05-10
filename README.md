# Pilot Design for ChannelNet 

This is an implementation of the paper "Pilot pattern design for deep learning based channel estimation in OFDM systems"

# Usage 

The main.py file shows an example of pilot pattern design for a VehA channel with 72 sub-carriers and 14 time slots. 
In this example, first indices are found using Concrete Autoencoder network and then the interpolation and SRCNN networks are trained based on the designed pilot setup. 

# Acknowledgement 
For implementation of the Concrete Autoencoder we have used the source codes provided by the authors: https://github.com/mfbalin/Concrete-Autoencoders

# Datasets 

links to access the data-sets:

Noisy channels as the input (including channels with all SNR vlaues reported in the paper):
https://drive.google.com/file/d/1w5H6jnD2jlf-sH5WmxyqVY-aVrjuVtvj/view?usp=sharing

Corresponding perfect channels as the output (channels without noise):
https://drive.google.com/file/d/19dH4sECa1Wo2O0uxQU2S1QjZJatU1WQY/view?usp=sharing



# Paper 
For more details see : https://ieeexplore.ieee.org/document/9166541
And please use the citation below : 

```
@ARTICLE{Soltani2020ChannelNet,
  author={M. {Soltani} and V. {Pourahmadi} and H. {Sheikhzadeh}},
  journal={IEEE Wireless Communications Letters}, 
  title={Pilot Pattern Design for Deep Learning-Based Channel Estimation in OFDM Systems}, 
  year={2020},
  volume={9},
  number={12},
  pages={2173-2176},
  doi={10.1109/LWC.2020.3016603}
  }
```




