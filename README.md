# nilm-dann
A repository for a semi-supervised approach for improving generalization in non-intrusive load monitoring

This repository corresponds to a technical solution titled _“A Semi-Supervised Approach for Improving Generalization in Non-Intrusive Load Monitoring”_, which can be found using the [link](https://www.pupin.rs/code/wp-content/uploads/2024/01/Semi-supervizirani-pristup-za-unapredjenje-generalizacije-kod-neintruzivnog-monitoringa-potrosnje-elektricne-energije.pdf).

A semi-supervised approach for improving generalization in non-intrusive load monitoring (NILM), described as a part of the corresponding scientific article and technical solution, is a novel approach for improving generalization performances for the problem of electrical energy consumption disaggregation on the appliance level is considered. The described domain adversarial neural network (DANN) approach improves disaggregation performances when data from the training domain differ from the data from testing domain, which is a common situation in practice.

In this repository, a trained neural network using the proposed DANN approach for disaggregation of microwave’s electrical consumption (```microwave_net.h5```) is provided, together with the used testing data set (```testing_dataset_microwave.pkl```). The script written in Python (```main.py```) is loading a pre-trained neural network-based model and is evaluating performance on the provided data set in consistence with the results presented in the aforementioned paper and technical solution.

The code is written in Python 3.11 and a list of requirements is given in ```requirements.txt```. Due to large file sizes for neural network and data set, it is necessary to clone the repository in an adequate manner using Git and Git LFS.
