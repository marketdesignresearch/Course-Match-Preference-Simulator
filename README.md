# Course Match Preference Simulator
This is a piece of software used for simulating students' preferences and their noisy reports in the Course Match mechanism. The structure of the students' preferences, as well as our proposed mechanism, are described the following paper:


**Machine Learning-powered Course Match**<br/>
Ermis Soumalias, Behnoosh Zamanlooy, Jakob Weissteiner, and Sven Seuken.<br/>
Full paper version including appendix: [[pdf](http://arxiv.org/abs/2210.00954)]

## Requirements

* Python 3.8 (or later) 
* numpy
* gurobi python API


Prepare your python environment <name_of_your_environment> (whether you do that with `conda`, `virtualenv`, etc.) and activate this environment. Then install the required packages.

Using conda:
```bash
$ conda install -c anaconda numpy 
$ conda install -c gurobi gurobi 

```


## Functionality
This simulator offers the following functionality


1. Generating the true students' preferences, for instances of any size (# number of students, # of courses, correlation of students' preferences etc.). Those preferences are described in Section 5 of our paper. 


2. Generating noisy students' reports, compatible with the Course Match GUI language. For 6 and 9 popular courses, we also provide parameters such that the resulting reported preferences match the findings regarding the accuracy of the students' reports from Budish & Kessler (2022).   

3. Determining a student's most preferred course schedule subject to feasibility constraints and her budget (with respect to her true or reported preferences).

4. Determining a student's true value for any bundle. 


## Demo 
The jupyter notebook provided showcases all of the above functionality. 

