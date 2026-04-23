# Network Systems And Design C400 Project
> Benji Kiblinger

## Comparison of Logistic Regression, Random Forest, and Convolutional Neural Networks in Classifying Encrypted and Non-Encrypted Network Traffic

> This project is an exploration of various types of models to classify both encrypted and non-encrypted network traffic data into various categories, such as chat, email, and p2p file sharing. In this project, I will be comparing a linear regression model as my baseline, to a random forest model, and finally, a convolutional neural network.

### Setup

To run this project, create a Python virtual environment and install the required dependencies:

```bash
python3 -m venv capstone_env
source capstone_env/bin/activate
pip install -r requirements.txt
```

Once the environment is active, you can run each model script individually:

```bash
python classification.py      # data exploration
```

### Dataset 

> The dataset I am using for this is the ISCXVPN2016 dataset, available from the University of New Brunswick. 
([Dataset](https://www.unb.ca//cic/datasets/vpn.html) )
