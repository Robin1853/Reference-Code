# Reference-Code
This Repository gives some introduction to my past coding experience.


### Masterthesis ###
My master's thesis is my largest project yet. I wrote additional code for analyzing the data and slight variations for complementary investigations, but only the most important part is shown here. QNN_1Layer_Pooling is the original code used in my thesis. For better structure and readability, I split it into different modules (main, qnodes, train_loop, quantum_classifier, plot_utils). Overall, 16 different quantum circuits were implemented in the network architecture. Each one was trained for 10 different initialization parameters, and over 10 epochs. The number of training images varies for the different datasets (5_7_MNIST: 400, Breastmnist: 546, MNIST: 10.000).
For the two datasets used, please look here: 
MNIST:    @article{lecun2010mnist,
           title={MNIST handwritten digit database},
           author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
           journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
           volume={2},
           year={2010}
          }
Breastmnist:   https://medmnist.com/
              @article{medmnistv2,
              title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
              author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
              journal={Scientific Data},
              volume={10},
              number={1},
              pages={41},
              year={2023},
              publisher={Nature Publishing Group UK London}
              }

### 2Layer_QNN ###
In the context of my master's thesis, I also conducted Investigations of two-layer quanvolutions in hybrid networks. The code is comparable to the Masterthesis code, but less intuitive, due to the self-coded hands-on approach, the reshaping of the images for the two-layer quantum circuits.

### Paper Crawler ###
A small hands-on code of mine for automatic paper search and storage of relevant papers on archive

