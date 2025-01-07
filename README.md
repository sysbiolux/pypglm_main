# PYPGLM (Probabilistic Graphical Logical Models)

## Description

PYPGLM package is designed to contexualize and model the DBN models corresponding to 
signaling and gene regulatory systems.
							 
PYPGLM is the new version of the toolbox "FALCON", developed in Matlab in 2017 and described in De Landtsheer et al. in 
Bioinformatics: [here is the link](https://academic.oup.com/bioinformatics/article/33/21/3431/3897376).

## Python Support

Python >=3.12 is required.


## Dependencies

1. networkx  
2. numpy  
3. pandas  
4. emoji  
5. pytest  
6. pytest-cov  
7. openpyxl  
8. scipy  
9. matplotlib  
10. gaft  
11. seaborn  
12. joblib  
13. scikit-learn  

## Installation Instructions

You can install `PYPGLM` directly from PyPI using pip:

```bash
pip install PYPGLM 
```



- You can also clone the repo to your local machine, set up a virtual environment,
install the requirements from the 'requirements.txt' file.


## Usage

To correctly model the network with its experimental data:
- The network topology file (.txt, .xls, .xlsx, .csv, and .sif) should contain the interactions of nodes within the investigated network, 
  specifying their type and gate.  
- Experimental data (xls, xlsx, csv) should consist of 3 tables representing the measured input and output node values and their 
  corresponding errors listed in columns across all experimental conditions listed successively in rows. Regarding the file in .csv 
  format, the 3 tables should be given as 3 individual files, while in the case of .xls(x) format, the 3 tables should 
  be presented as 3 distinct sheets within the same corresponding file.

Examples demonstrating how to run the optimization and regularization are provided in 'Driver_PYPGLM_Optimization' and 
'Driver_PYPGLM_Regularization' files, respectively."

## Optimization

The optimization algorithms within PYPGLM are 'SLSQP', 'L-BFGS-B', and 'Trust-constr'.



## Regularizations

The regularization algorithm within PYPGLM includes:
- L1: simple L1 norm on parameters. This induces a pruning of less important inhibitory edges, but has no effect on stimulatory edges
- L2: Induces a decrease of weights of unimportant inhibitory edges
- L1_groups: Induces an equalization of edge values between contexts when these are discrete (different cell lines)

## Credit

Developers:

Salma Bayoumi,                salma.ismail.hamed@gmail.com
				              salma.bayoumi.001@student.uni.lu
							  
Sebastien De Landtsheer,      sebastien.delandtsheer@uni.lu
                              seb@delandtsheer.com



