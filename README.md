# Adversarial Environment Generation for Multi-Agent Algorithm Evaluation
Authors: Kavya Kopparapu, Eric Lin, and Lucy Liu  

Advised by Prof. Radhika Nagpal  
Harvard CS 289, Fall 2021  

Our project is the first known implementation of generating adversarial environments for multi-agent programs using a genetic algorithm (GA). We implement from scratch an environment simulator to test multiple foraging algorithms (including deep Q learning and biologically-inspired swarm algorithms) against our proposed GA. We also demonstrate significant improvement from co-evolving these multi-agent and genetic algorithms together.  

This repo contains the code, figures, and saved (pickled) simulations from experiments that accompany our work. The following is a breakdown of the folder structure:

* Environment simulator:
    * `environment.py` sets up the environment model and classes
    * `utils.py` contains necessary helper functions
    * `geneticalg.py` contains implementation of the genetic algorithm for adversarial environment generation
    * `agent.py` contains classes of different agents. Currently this includes random walk, swarm, and deep q learning. This is modularized to be easily extended to other agent algorithms.
* Code for running experiments:
    * `run_ga.py` runs the genetic algorithm
    * `run_coevolution.py` runs the co-evolution between the genetic algorithm and deep q learning algorithm
    * `run_coevolution_evaluation.py` contains code for co-evolution evaluation
* Other root directory code files:
    * `train_dqn.py` contains code for training the deep Q learning model. It requires a GPU, but you can toggle this with cuda settings.
* `DQN` folder: 
    * `target_net.pt` contains our saved trained deep Q learning model for foraging.
* `Results` folder:
    * `Final` folder contains the pickled outputs from our final experiments runs. These outputs include the experiment parameters, stored fitness values, and grids (for GIF visualizations) for each of the 50 generations in the simulation. There are over 50 experiment runs with different parameters that you can see in its name. 
    * `pickle_processor.ipynb` and `co_evolution_processor.ipynb` are jupyter notebook scripts that process the data output from the pickle outputs in the  `Final` folder.
* `GIFs` folder:
    * show visualizations of the agents foraging food for 300 time steps on a variety of different environments.
