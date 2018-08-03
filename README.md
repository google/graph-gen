This is not an official Google Product. 

Code for NVP model over Graphs. Only Preliminary results yet. Thanks Yujia Li for providing the code for parsing data files.

Requirements:\
	- Tensorflow (Please see [http://www.tensorflow.org] for how to install/upgrade)\
	- Numpy (Please see [http://www.numpy.org])\
	- Tensor2Tensor (Please see [https://github.com/tensorflow/tensor2tensor] for how to install Tensor2Tensor)\
	- Scikit-Learn (Please see [http://scikit-learn.org/stable/developers/advanced_installation.html] for more details)

Download the Processed versions of the Zinc and ChEMBL datasets from the links below. Again, Credits to Yujia Li for helping us with these datasets.\
	- ChEMBL: ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_23/chembl_23_chemreps.txt.gz\
	- Zinc: https://github.com/mkusner/grammarVAE/tree/master/data

In order to run the model, download the data from the sources provided above and run it as
`python trainer.py --exp_dir=<directory for experiments> --batch_size=100 --dataset=<zinc/max20> --train_steps=50000 --sample=<True/False whether run sampling or training> --use_BN=<True/False> --only_nodes=False --perturb_data=<True/False> --use_edge_features=<True/False>`\

Some important points to take care of:
1. Please set --sample=True, if you want to start from some samples in the latent space and transform them into actual graphs.
2. Perturbing data in the data space can be done by setting the flag --perturb_data=True.
3. The code currently defaults to using beta prior for the adjcacency matrix as well as the node features.
4. Use the flag --use_edge_features=True to indicate that you also want to generate edge features apart from the adjacency matrix.

Maintained by Kevin Swersky. Work done by Aviral Kumar, Kevin Swersky (kswersky@google.com) and Jimmy Ba (jimmylb@google.com). For any questions, please contact Kevin Swersky (Github username: kswersky)
