# Neural Network for Predicting Collsion
This is a demonstration of collision-detection and driving using a feedforward neural network from Interactive Robotics Lab at Arizona State University.


## Requirements

For smooth installation on Windows, Linux and Mac. It is recommended to install anaconda from here https://www.anaconda.com/download/

After installing Anaconda, Please run the following commands.

```

conda create -name intel_game_env python=3.6
source activate intel_game_env   ## Windows -> conda activate intel_game_env 
conda install pytorch-cpu -c pytorch 

pip install cython
pip install torchvision
pip install pygame
pip install pymunk


```

## Usage

To run this project. Please do the following steps

```python command

python training_the_model.py

```
It opens up the simulator. The bot drives around randomly, sometimes bumping into the walls. All the sensor data during this simulation is collected and stored in 'sensor_data.txt'

```python
python make_it_learn.py
```

This program does three things:
 
1. Loads the sensor data collected and labels all the collision data as 1 and the rest of them as 0.
2. Creates a feedforward neural network and trains with labeled data up to 25 epochs. 
3. Stores the trained model as 'nn_bot_model.pkl'


```python
python playing_the_model.py
```
This program loads the neural network model and opens up the simulator. The bot is programmed to drive itself to the destination and feeding the sensor data to the neural network at everytime step.
If the neural network detects collision bot turns green and takes alternative action, to the action it was planning to take.

### Optional

After running the above commands, to have different start position; run any of the commands below

```python
python playing_the_model.py 1
```
```python
python playing_the_model.py 2
```
```python
python playing_the_model.py 3
```
```
```

## Feedback

Questions or comments may be directed to Nambi Srivatsav at <nsrivats@asu.edu>
