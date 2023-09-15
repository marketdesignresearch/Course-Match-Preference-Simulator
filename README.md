# Course-Match-Preference-Simulator
A clean repository of the Preference Simulator component of our Machine Learning-powered Course Match paper. 

Arxiv: https://arxiv.org/abs/2210.00954

## Requirements

* Python 3.8 (or later) 
* numpy
* gurobi python API


## Dependencies

Prepare your python environment <name_of_your_environment> (whether you do that with `conda`, `virtualenv`, etc.) and activate this environment. Then install the required packages

Using conda:
```bash
$ conda install -c anaconda numpy 
$ conda install -c gurobi gurobi 

```


## Functionality
This simulator offers the following functionality


### 1. Generating the true students' preferences, for instances of any size (# number of students, # of courses, correlation of students' preferences etc.). Those preferences are described in Section 5 of our paper. 


### 2. Generating noisy students' reports, compatible with the Course Match GUI language. For 6 and 9 popular courses, we also provide parameters such that the resulting reported preferences match the Budish and Kessler findings regarding the accuracy of the students' reports. 

### 3. Determining a student's most preferred bundle subject to feasibility constraints and her budget, with respect to both her true preferences, and her reported ones. 

### 4. Determining a student's true value for any bundle. 

