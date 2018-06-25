
This is a demonstration of collision-detection and driving using a feedforward neural network from Interactive Robotics Lab at Arizona State University.


## Requirements

Please install Pygame and PyMunk in Python3 environment.

```python
pip install pygame
pip install pymunk
```

## Usage

To run this project. Please do the following steps

For collecting the collision data for training:
```python
python training_the_model.py
```

For training the neural network:
```python
python make_it_learn.py
```

For running the project which learned to take right actions avoiding collisions:
```python
python playing_the_model.py
```
Car learns to avoid collision but by default it moves to towards the destination.



## Feedback

Questions or comments may be directed to Nambi Srivatsav at <nsrivats@asu.edu>
