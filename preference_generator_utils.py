from gurobipy import GRB
import gurobipy as gp
import random
# from time import process_time
from timeit import default_timer as timer
import numpy as np

import functools
import operator
from collections import Counter
import math

import os
import sys
import inspect
import pickle 

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


module_path = os.path.abspath(os.path.join('./MIP/'))
if module_path not in sys.path:
    sys.path.append(module_path)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb+') as f:
        return pickle.load(f)

def courses(n_types = 1, n_courses = 20, n_A_pop = 5, n_A_npop = 10, seed = 0):
    """
    Return the popular and unpopular courses
    Input:

    n_types: int
    number of student types in the problem

    n_courses: int
    number of courses in the problem

    n_A_pop: int
    number of popular courses that the students' popular courses will be drawn from

    n_A_npop: int or list of int
    number of unpopular courses that the students' popular courses will be drawn from

    Output:
    course_dict: dictionary
    keys: range(0, n_types) for student type
    return A_pop, A_npop, A_0
    """
    random.seed(seed)
    course_dict = dict()
    courses = np.array(range(0, n_courses))

    # if the user wants the n_A_pop to be the same for all types of students they enter an integer instead of list
    # the int is converted to a list of size of student types
    if isinstance(n_A_pop, int):
        n_A_pop = [n_A_pop]*n_types
    if isinstance(n_A_npop, int):
        n_A_npop = [n_A_npop]*n_types
    for i in range(n_types):
        courses_idx = list(range(n_courses))
        # uniformly at random smaple the courses
        pop_idx = random.sample(courses_idx, n_A_pop[i])
        pop = courses[pop_idx]
        #delete the courses that were chosen as popular
        for j in sorted(pop_idx, reverse=True):
            del courses_idx[j]
        npop_idx = random.sample(courses_idx, n_A_npop[i])
        npop = courses[npop_idx]
        #define the A_0 set
        zero = np.delete(courses, pop_idx + npop_idx)
        course_dict[i] = {'A_pop': pop,
                      'A_npop': npop,
                       'A_0': zero}
    return course_dict


#   courses(n_types = 1, n_courses = 20, n_A_pop = 5, n_A_npop = 10, seed = 0)
def draw_favorites(pop = [1, 2, 3], lpop = [4, 5, 6], n = 3, p = 0.9, seed = 0):
    """
    This function draws n items from the lists pop+lpop non-uniformly with probability p from pop and probability 1-p from lpop

    Input:
    pop: list or np.array
        The list of more popular courses
    lpop: list or np.array
        The list of less popular courses
    n: int
        number of courses
    p: float
       the probability associated with the list pop
    seed: int
        seed

    Output:
    selected : np.array
        seelcted list of courses
    pop_cur: np.array
         courses that were not selected from pop+lpop --> pop+lpop-selecte
    """
    np.random.seed(seed)
    l = list(pop) + list(lpop)

    # normalize probabilities so they add up to one (requirement for np.random.choice)
    t = p * len(pop) + (1-p) * len(lpop)
    p_1 = p/t
    p_2 = (1-p)/t
    p = [p_1]*len(pop) + [p_2]*len(lpop)

    pop_idx = np.random.choice(a = range(len(l)), size = n, p=p, replace = False)

    selected = np.array(l)[pop_idx]
    pop_cur = np.delete(l, pop_idx)

    return selected, pop_cur


def draw_pop_unpop(A_pop, A_npop, A_0, n_pop, n_npop, p_pop, p_npop, seed):
    """
    The function draws popular, unpopular, and value zero courses for a specific student

    Input:
    A_pop
    A_npop
    A_0

    n_pop: int
        number of popular courses for student i
    n_npop: int
        number of unpopular courses for student i
    p_pop: float
        The probability that the popular course is chosen from A_pop
    p_npop: float
        The probability that the unpopular course is chosen from the updated A_npop = A_pop + A_npop - i_pop
    seed: int
        seed

    ouuput:
    i_pop : list
        list of the student's popular courses
    i_npop: list
        list of the student's unpopular courses
    i_0: list
        value zero courses for student i
    """
    i_pop, A_npop = draw_favorites(A_pop, A_npop, n_pop, p_pop, seed)
    i_npop, i_0 = draw_favorites(A_npop, A_0, n_npop, p_npop, seed + 100)

    return i_pop, i_npop, i_0


def centers(i_pop = [1, 2, 3], i_npop = [4, 5, 6], n = 2, p_pop = 0.9, seed = 0):
    """
    This function returns the centers for calculating the complementarity and substitution sets for each student

    Input:
    i_pop: list or np.array
        The list student i's popular courses

    i_npop: list or np.array
        The list of student i's unpopular courses

    n: int
        the number of centers

    p_pop: float in [0,1]
        the relative probability that the centers are drawn from i_pop

    output:
        list of centers
    """
    return draw_favorites(i_pop, i_npop, n, p_pop, seed)


def make_grid(n_courses = 20, height = 4, width = 5, seed = 0):
    """
    Input:
    Courses: dictionary with keys pop and npop
             correponding to the student's popular and unpopular courses
    d_c: int or list of int
         the distnace up until whicht the courses are assumed to be complementary with the popular courses
    d_s: int or list of int
         the courses between d_c and d_s are considered to be in the set of suctitutes for popular courses

    output:
    set_c: list of lists
           complementarity lists
    set_s: list of lists
            substitution lists
    """

    x_y = []
    for i in range(height):
        for j in range(width):
            x_y.append((i+1, j+1))
    # randomly sample without replacement from the x_y grid
    return random.sample(x_y, n_courses)


def perturb_grid(grid, seed):
    # Later
    return grid


def manhattan(a = [3, 4], b = [5, 6]):
    """
    Input:
    a : list
    b: list

    output:
    Manhattan ditance between a and b
    """
    return sum(abs(val1-val2) for val1, val2 in zip(a, b))


def chebyshev(a = [3, 4], b = [5, 6]):
    """
    Input:
    a : list
    b: list

    output:
    chebyshev or l_0 ditance between a and b
    """
    return max(abs(v1 - v2) for v1, v2 in zip(a, b))


def rm_duplicates(l = [1, 1, 3, 3]):
    new_l = []
    for elem in l:
        if elem not in new_l:
            new_l.append(elem)
    return new_l


def sub_comp(grid, cntr, viable_courses = [1, 2, 3, 4, 5, 6], d_s = 1, d_c = 2, d_l = 2, seed = 0):
    """
    The function that calculates the compleentarity/substitution set

    Input:

    grid: list of tuples
    The location of each course on the grid

    cnt: list
    list of courses that are used as centres

    viable_courses: list
    courses that have non zero valeu : i_pop + i_npop

    d_s: int or list of int
        manhattan distance for substitutes
    d_c: int or list of int
        manhattan distance for complementarities
    d_l: int or list of ints
        l_0 distance outside which there is no sub/comp relationship
    seed: int
        seed

    output:

    set_c: list of lists
        each list is a complimentarity set

    set_cs: list of lists
        each list is a substitution set

    """
    if isinstance(d_c, int):
        d_c = [d_c]*len(cntr)

    if isinstance(d_s, int):
        d_s = [d_s]*len(cntr)

    if isinstance(d_l, int):
        d_l = [d_l]*len(cntr)

    d_dict = dict()
    set_c = []
    set_s = []
    for i, c in enumerate(cntr):
        list_c = [c]
        list_s = []

        # courses start from 1
        for j in viable_courses:
            m_cur = manhattan(grid[c], grid[j])
            ch_cur = chebyshev(grid[c], grid[j])

            # check wether the course belongs to a comp/sub set

            if m_cur <= d_s[i]:
                list_s.append(j)
            elif m_cur > d_s[i] and ch_cur <= d_c[i]:
                list_c.append(j)
        # if the center is only in the list drop it
        if len(list_c) > 1:
            set_c.append(list_c)
        if len(list_s) > 1:
            set_s.append(list_s)

    return rm_duplicates(set_c), rm_duplicates(set_s)


def base_values(n_courses = 20, i_pop = [1, 2, 3], i_npop = [4, 5, 6], pop_low = 80, pop_high = 120, npop_low = 40, npop_high = 60, seed = 0):
    """
    The function is meant to generate base values for non-zero values courses

    input:

    n_courses: int
        number of courses
    i_pop: list
         list of popular courses fo student
    i_npop: list
         list of inpopular student
    pop_low, pop_high, npop_low, npop_high: int
    the ranges indicating the random uniform generation for values of popular and unpopoular course

    output: list
        base value of all courses
    """
    np.random.seed(seed)
    pop_base = np.random.uniform(pop_low, pop_high, len(i_pop))
    npop_base = np.random.uniform(npop_low, npop_high, len(i_npop))

    base = np.zeros(n_courses)
    for j, _ in enumerate(pop_base):
        base[i_pop[j]] = pop_base[j]
    for j, _ in enumerate(npop_base):
        base[i_npop[j]] = npop_base[j]
    return base

# def comp_func(n , seed = 0, expected_value = 0.2):
#     np.random.seed(seed)
#     return np.cumsum([0] + list(np.random.uniform(0, 2 * expected_value, n - 1)))

# def sub_func(n, seed = 0, expected_value = 0.2):
#     np.random.seed(seed)
#     return np.cumsum([0] + list(np.random.uniform(- (2 * expected_value),0, n - 1)))


def comp_func(n, expected_value = 0.2, is_fixed = True, decay = 0.5, seed = 42):
    """
    n: int
        Number of items
    expected_value: float
        returns to the old thing with expected values
    is_fixed: Bool
        If true: No distribution
    decay = float
        How fast the value decays/grows with the number of items from the set
    """
    rng = np.random.default_rng(seed)
    decay_list = []
    for i in range(0, n-1):
        decay_list = decay_list + [decay**i]
    if is_fixed:
        mult_base = np.repeat(expected_value, n - 1)*np.array(decay_list)
    else:
        mult_base = rng.uniform(0, 2 * expected_value, n - 1) * np.array(decay_list)
    mult_base = [0] + list(mult_base)
    return np.cumsum(mult_base)


def sub_func(n, expected_value = 0.2, is_fixed = True, decay = 0.5, seed = 42):
    """
    n: int
        Number of items
    expected_value: float
        returns to the old thing with expected values
    is_fixed: Bool
        If true: No distribution
    decay = float
        How fast the value decays/grows with the number of items from the set
    """
    rng = np.random.default_rng(seed)
    decay_list = []
    for i in range(0, n-1):
        decay_list = decay_list + [decay**i]
    if is_fixed:
        mult_base = np.repeat(-expected_value, n - 1)*np.array(decay_list)
    else:
        mult_base = rng.uniform(- 2 * expected_value, 0, n - 1) * np.array(decay_list)
    mult_base = [0] + list(mult_base)
    return np.cumsum(mult_base)


def student_preference_simluator(courses, grid, type_i = 0, n_i_pop = 3, n_i_npop = 5, i_p_pop = 0.9, i_p_npop = 0.9, n_cntr = 2,
            i_p_pop_cntr = 0.9, d_s_i = 1, d_c_i = 2, d_l_i = 2, pop_low = 80, pop_high = 100, npop_low = 60, npop_high = 80,
            complements_expected_value = 0.2, substitutes_expected_value = 0.2, seed_i = 0, fixed_comp_values = True, complements_decay = 0.5, substitutes_decay = 0.3):
    """
    This function will generate the base_values and complementarity and substitution sets for a single student of type type_i

    Input:
    courses: dictionary
    The output of the courses() function e.g : {0: {'A_pop': array([12, 13,  1,  8, 15]),
                                                    'A_npop': array([ 7, 17,  5,  9,  6, 11,  4,  3, 16,  2]),
                                                    'A_0': array([ 0, 10, 14, 18, 19])}}
    grid: list of tuples
          coordinates of each course on the xy map

    type_i: int
        the student's type, corresponds to the keys in the courses

    n_i_pop: int
        number of favorite courses for the student to be generated

    n_i_npop: int
        number of non-favorite courses for the student to be generated

    i_p_pop: float in [0,1]
        the probability that the favorite vourses are drawn from A_pop

    i_p_npop: float in [0,1]
        the probability that the non favorite  courses are slected from A_npop

    n_cntr: int
        number of courses to be used as the center or in other words number of complementarity sets = number of substitution sets

    i_p_pop_cntr: float in [0,1]
        The probability that the center is chosen from the students popular courses

    d_s_i: int or list of int
           manhattan distance for substitutes

    d_c_i: int or list of int
         manhattan distance for complementarities

    d_l_i: int or list of ints
         l_0 distance outside which there is no sub/comp relationship

    pop_low, pop_high, npop_low, npop_high: int
    the ranges indicating the random uniform generation for values of favorite and unfavorite courses

    seed_i: int
        the seed controlling the genration of favorite, unfavorite courses as well as the centers and the values for courses
    fixed_comp_values: Bool
        If true: the comp values are set to their expected value instead of being drawn form a uniform distribution
    """
    c = courses[type_i]
    A_pop = c['A_pop']
    A_npop = c['A_npop']
    A_0 = c['A_0']

    # generate favorite and non-favorite courses for the student
    i_pop, i_npop, i_0 = draw_pop_unpop(A_pop, A_npop, A_0, n_i_pop, n_i_npop, i_p_pop, i_p_npop, seed_i)
    gr = perturb_grid(grid, seed_i)
    cnt, _ = centers(i_pop, i_npop, n_cntr, i_p_pop_cntr, seed_i)
    C, S = sub_comp(gr, cnt, np.append(i_pop, i_npop), d_s_i, d_c_i, d_l_i, seed_i)

    list_S = []
    # get the marginal values for having an additional course in the substitution list
    for i in range(len(S)):
        list_S = list_S + [(S[i], sub_func(n = len(S[i]), seed = seed_i + i, expected_value = substitutes_expected_value, is_fixed = fixed_comp_values, decay = substitutes_decay))]

    list_C = []
    for i in range(len(C)):
        list_C = list_C + [(C[i], comp_func(n = len(C[i]), seed = seed_i + i, expected_value = complements_expected_value, is_fixed = fixed_comp_values, decay = complements_decay))]

    # generate the base values for the courses
    base = base_values(len(A_pop) + len(A_npop) + len(A_0), i_pop, i_npop, pop_low, pop_high, npop_low, npop_high, seed_i)
    return base, list_S, list_C, i_pop, i_npop


# -----  useful_functions.ipynb code -----

#   timetable generator:

def timetable_generator(n_courses = 30, days = 5, timeslots = 12, timeslot_probabilities = {1: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05],
                        2: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.1, 0.0], 3: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.15, 0.0, 0.0]},
                        credit_units = [0.5 for i in range(15)] + [1 for i in range(15)],
                        hour_configurations = {0.5: [[2, 1], [2]], 1: [[3, 2],  [2, 2]]}, seed = 42):
    """
    A function generating the timetable of all the courses

    Parameters:
    ----------------------------------
    n_courses : int
        The number of courses
    days : int
        The number of school days (e.g. 5)
    timeslots : int
        The number of time slots per day (e.g. 12)
    timeslot_probabilities: dictionary from "hours" to list of reals of size t
        The i-th element of each list is the probability of a course being taught in the i-th timeslot of a day (morning time slots are more likely than evening ones)
        The reason for the "dictionary" is because it is likely that 3-hour courses are taught later than 2-hour ones, for example.
    credit_units: list of length n of the form [0.5, 1, 0.5, ...]
        The i-th element of that list is the number of credit units of the i-th course
    hour_configurations: dictionary from credit units -> possible time configurations for a course with that amount of credit units
        e.g. h[1] = [ [2,2], [3,1] ] a 1-CU course can either have 2 2-hour slots in the schedule or one 3-hour and one single-hour one


    Returns:
    ----------------------------------
    C: 3-D array of shape (d, t, )
        C[i][j]: The list of all courses being taught in the j-th time slot of the i-th day
    """

    if (seed is not None):
        random.seed(seed)
        np.random.seed(seed)

    # initialize a schedule with all the time slots empty
    timetable = []
    for i in range(days):
        day = []
        for j in range(timeslots):
            day.append([])
        timetable.append(day)

    for i in range(n_courses):
        #         print("--- /// ---")
        #         print("course number: ", i)
        hour_configuration = random.sample(hour_configurations[credit_units[i]], 1)[0]   # get an hour configuration for the i-th course, based on its credit units
        course_days = random.sample(range(0, days), len(hour_configuration))  # get a different weekday for every of the different hours of that course e.g. if a course is 3 hours in a day and 1 in another, get 2 days
#         print("hour configuration: ", hour_configuration)
        for j in range(len(hour_configuration)):
            lecture_duration = hour_configuration[j]
#             print("timeslots: ", timeslots)
#             print("lecture duration: " , lecture_duration)
#             print("timeslot probabilities for that lecture duration: ", timeslot_probabilities[lecture_duration])
            start_timeslot = np.random.choice(timeslots, p = timeslot_probabilities[lecture_duration])  # e.g. a 2 hour lecture cannot start at the final timeslot
            for k in range(lecture_duration):
                timetable[course_days[j]][start_timeslot + k].append(i)    # add the course on the "start timeslot" plus the following ones, depending on that course's hours

    return timetable


# time preferences generator:

def time_pref_generator(days = 5,  student_types = ['no_overload', 'free_days', 'few_timegaps', 'balanced'], type_probabilities = [1 / 4 for i in range(4)],
                        overload_params = {'no_overload': (-20, 8), 'free_days': (-7, 5), 'few_timegaps': (-7, 5), 'balanced': (-10, 5)},
                        timegap_penalty_params ={'no_overload': (-4, 2), 'free_days': (-4, 2), 'few_timegaps': (-10, 4), 'balanced': (-7, 3)},
                        free_days_params = {'no_overload': ([40, 24, 10, 0, 0], [12, 8, 4, 0, 0]), 'free_days': ([60, 40, 20, 0, 0], [14, 6, 6, 0, 0]), 'few_timegaps': ([40, 24, 10, 0, 0], [12, 8, 4, 0, 0]), 'balanced': ([50, 30, 10, 0, 0], [13, 8, 4, 0, 0])},
                        seed = 42, disable_time_prefs = True):
    """
    A function generating the time preferences of a single student.

    Parameters:
    ----------------------------------
    days: int
        The number of school days (e.g. 5)
    timeslots: int
        The number of time slots per day (e.g. 12)
    student_types: list of strings
        student_type[i]: the i-th possible student type, e.g. ``balanced'' for a student that prefers a balanced timeschedule

    seed: int or None
        The seed to be used for all randomness


    Returns:
    ----------------------------------
    overload_penalty: float
        optimal_schedule[i] = 1 if course x_i is contained in the optimal bundle for that student, 0 else
    timegap_penalty: float
        The total time of the function call, in seconds. Only returns it if time_output = True.
    free_days_marginal_values: list of floats
    """

    if(disable_time_prefs):
        return (0, 0, [0, 0, 0, 0, 0])

    if(type(seed) != type(None)):
        random.seed(seed)
        np.random.seed(seed)

    student_type = np.random.choice(student_types, p = type_probabilities)  # get the student type
#     print(student_type)

    overload_penalty = random.randint(math.floor(overload_params[student_type][0] - (overload_params[student_type][1]/2)), math.ceil(overload_params[student_type][0] + (overload_params[student_type][1]/2)))
    timegap_penalty = random.randint(math.floor(timegap_penalty_params[student_type][0] - (timegap_penalty_params[student_type][1]/2)), math.ceil(timegap_penalty_params[student_type][0] + (timegap_penalty_params[student_type][1]/2)))

    free_days_marginal_values = []
    for i in range(days):
        free_days_marginal_values.append(random.randint(math.floor(free_days_params[student_type][0][i] - (free_days_params[student_type][1][i] / 2)), math.ceil((free_days_params[student_type][0][i] + (free_days_params[student_type][1][i] / 2)))))

    return(overload_penalty, timegap_penalty, free_days_marginal_values)


# calculate bundle value by hand:

def calculate_bundle_value_by_hand(bundle, course_timetable, credit_units, additive_preferences,  complements_sets = [], substitutes_sets = [], max_bundle_size = 6,
                  overload_penalty = -15, timegap_penalty = -10, free_days_marginal_values = [40, 30, 10, 1, 1], days = 5, timeslots = 12, base_perturbation_distribution = None, base_perturbation_spread = 0.1,
                                   complements_forget_probability = None, substitutes_forget_probability = None, seed = None,  ignore_timegaps = False,  verbose = True, print_solution = False, time_output = False):
    """
    A function that for a given (legal) bundle and student returns either the student's true value for that bundle or the value after preference-reporting errors, provided there are no timegaps, without solving any MIPs.

    Parameters:
    ----------------------------------
    bundle: 0-1 list
        bundle[i] = 1 if the bundle contains the i-th course, 0 if it does not
    course_timetable : 3-D Array of shape (days, timeslots, )
        course_timetable[d][t]: A list of all the courses with a lecture on timeslot t of day d
    credit_untis: 1-D array of shape (n_courses)
        credit_units[i]: The credit units of the i-th course
    additive_preferences: list of floats
        additive_preferences[i]: The student's additive value for the i-th course
    complements_sets: list of tuples
        each tuple has as its first element the list of items belonging in that set of complements, and as its second element the complement function psi' of that bundle,
        e.g. complements_sets[0] = ([2,5,7,17,35], [0, 0.2, 0.4, 0.6, 0.8])
    substitutes_sets: list of tuples
        each tuple has as its first element the list of items belonging in that set of substitutes, and as its second element the substitute function xi of that bundle,
        e.g. substitutes_sets[0] = ([2,5,7,17,35], [0, -0.2, -0.4, -0.6, -0.8 ])
    max_bundle_size: int
        The maximum number of courses that the student can take in a semster
    overload_penalty: float
        How much negative utility the student has for having overloaded days
    timegap_penalty: float
        How much negative utility the student loses for each timegap he has to spend at school
    free_days_marginal_values: list of floats
        free_days_marginal_values[i]: The exta utility for having an (i+1)-th free day, provided the student already has i free days.
    already_taken: list of ints
        The list of all the courses that that student has already taken
    days : int
        The number of school days (e.g. 5)
    timeslots : int
        The number of time slots per day (e.g. 12)
    base_perturbation_distribution: string or None
        The distribution by which the actual bundle value will be perturbed (to simulate error reporting). Options: None, 'Uniform', 'Gaussian'
    base_perturbation_spread: float
        The spread of the base perturbation distribution.
    complements_forget_probability: float
        The probability that a student has to "forget" that a specific course belongs in a complement set.
    substitutes_forget_probability: float
        The probability that a student has to "forget" that a specific course belongs in a substitute set.
    seed: int or None
        The random seed used
    ignore_timegaps: Boolean
        If true: The timegap part of the objective and constraints are removed from the objective, improves the time performance of the MIP, as those are very costly
    verbose: boolean
        If true: The function writes the corresponding MIP to the file StudentLP.lp
    print_solution: boolean
        If true: The function prints in an easy to read format all binary variables set to 1 for debug purposes
    time_output: boolean
        If true: The function instead of returning the optimal schedule returns the tuple (optimal schedule, total_time)

    Returns:
    ----------------------------------
    bundle_value: float
        The student's value for that bundle.
    total_time: float
        The total time of the function call, in seconds. Only returns it if time_output = True.
    """

    start = timer()
    bundle_value = 0

    if (seed):
        random.seed(seed)
        np.random.seed(seed)

    # add the base values part of the objective
    for i in range(len(additive_preferences)):
        bundle_value = bundle_value + bundle[i] * additive_preferences[i] * credit_units[i]

    # add the complements/substitutes part of the objective
    for (complement_set, complement_function) in complements_sets:
        complement_set_courses = 0
        for j in complement_set:   # count the number of courses in the bundle that belong in the complement set
            complement_set_courses = complement_set_courses + bundle[j]
        if (not complements_forget_probability):
            for j in complement_set:
                bundle_value = bundle_value + bundle[j] * additive_preferences[j] * credit_units[j] * complement_function[max(0, complement_set_courses - 1)]
            # add to the value the value of every course in that complement set times the "bonus function" for having that number of courses in that set
        else:
            for j in complement_set:
                if(random.uniform(0, 1) > complements_forget_probability):  # the chance of the condition being true == 1 - complements_forget_probability
                    bundle_value = bundle_value + bundle[j] * additive_preferences[j] * credit_units[j] * complement_function[max(0, complement_set_courses - 1)]
            # in case there is a chance of forgetting a bundle, you only count its value if you did not forget it

    for (substitute_set, substitute_function) in substitutes_sets:
        substitute_set_courses = 0
        for j in substitute_set:
            substitute_set_courses = substitute_set_courses + bundle[j]
        if (not substitutes_forget_probability):
            for j in substitute_set:
                bundle_value = bundle_value + bundle[j] * additive_preferences[j] * credit_units[j] * substitute_function[max(0, substitute_set_courses - 1)]
        else:
            for j in substitute_set:
                if(random.uniform(0, 1) > substitutes_forget_probability):
                    bundle_value = bundle_value + bundle[j] * additive_preferences[j] * credit_units[j] * substitute_function[max(0, substitute_set_courses - 1)]

    # calculate the number of school hours per day
    hours_per_day = []
    for day in course_timetable:
        hours_on_that_day = 0
        for hour in day:
            for j in hour:
                hours_on_that_day = hours_on_that_day + bundle[j]
                # for every course that is tought on every hour of the day, increase the hours_on_that_day by 1 if the student has actually taken that course
        hours_per_day.append(hours_on_that_day)

    free_days = len([i for i in hours_per_day if i == 0])  # say hi to python for me

    for i in range(0, free_days):
        bundle_value = bundle_value + free_days_marginal_values[i]

    overloaded_day_hours = max(hours_per_day)

    avg_other_days = (sum(hours_per_day) - max(hours_per_day)) / (len(hours_per_day) - 1)
    overload = overloaded_day_hours - avg_other_days

    bundle_value += overload * overload_penalty

    # perturb the bundle value, if a perturbation distribution has been specified
    if(base_perturbation_distribution == 'Gaussian'):
        alpha = np.random.normal(loc = 1, scale = base_perturbation_spread)
        bundle_value = alpha * bundle_value

    elif(base_perturbation_distribution == 'Uniform'):
        alpha = random.uniform(1 - (base_perturbation_spread/2), 1 + (base_perturbation_spread/2))
        bundle_value = alpha * bundle_value

    end = timer()
    if (time_output):
        return(bundle_value, end-start)

    return bundle_value


# Generate student preferences

def generate_student_preferences(course_buckets, course_grid,  thematic_types = [0], thematic_type_probabilities = [1], time_preference_type_probabilities = [1 / 4 for i in range(4)], days = 5, timeslots = 12,
                                favourite_courses_number = [2], favourite_courses_probabilities = [1], non_favourite_courses_number = [10 + i for i in range(5)], non_favourite_courses_probabilities = [1/5 for i in range(5)],
                                favourite_pop_probability = 1, non_favourite_npop_probability = 1, center_numbers = [2], center_number_probabilities = [1], favourite_center_probability = 1,
                                substitute_range = 1, complement_range = 2, complements_expected_value = 0.2, substitutes_expected_value = 0.2,
                                d_l_i = 10, favourites_value_range = (80, 120), non_favourites_value_range = (40, 60), complements_decay = 0.5, substitutes_decay = 0.3,
                                 disable_time_prefs = True, fixed_comp_values = True, seed = None):
    """
    A function that creates the full preferences of a student.

    Parameters:
    ----------------------------------
    course_buckets: dictionary of dictionaries
        For every type of student, it returns a dicitonary with the A_pop, A_npop and A_0 courses for that type of student
    course_grid: list of (int,int) tuples
        course_grid[i]: The (x,y) coordinates of the i-th course
    thematic_types: list
        contains all the possible student types
    thematic_type_probabilities: list of floats
        thematic_type_probabilities[i]: The probability of a student actually being of the i-th thematic type
    time_preference_type_probabilities: list of floats
        time_preference_type_probabilities; The probability of a student having the i-th time preference type
    days: int
        The number of school days per week
    timeslots: int
        The number of slots in the course timetable per day
    favourite_courses_number: list of ints
        Contains all possible numbers of favourite courses that a student may have
    favourite_courses_probabilities: list of floats
        favourite_courses_probabilities[i]: The probability of a student actually having favourite_courses_number[i] favourite courses
    non_favourite_courses_number: list of ints
        Contains all possible numbers of non-favourite (but non-zero) courses that a student may have
    non_favourite_courses_probabilities: list of floats
        non_favourite_courses_probabilities[i]: The probability of a student actually having non_favourite_courses_number[i] non favourite courses
    favourite_pop_probability: float
        The relative probability of a favourite course for the student to be drawn from the bucket of the A_pop courses, and with 1 - that probability
        the course is drawn from the A_npop bucket
    non_favourite_npop_probability: float
        The relative probability of a non-favourite non-zero course to be drawn from the A_npop bucket of courses, and with 1 - that probability the
        course is drawn from the A_0 bucket
    center_numbers: list of ints
        The possible number of complement/substitute centers that a student may have
    center_number_probabilities: list of floats
        center_number_probabilities[i]: The probability of a student having center_numbers[i] centers.
    favourite_center_probability: float
        The relative probability that the center of a student's complement/substitute set will be around one of her favourite courses
    substitute_range: int
        The manhattan distance from the center of a circle for which non-zero courses are considered substitutes
    complements_range: int
        The manhattan distance from the center of a circle for which non-zero courses that are not substitutes are considered complements
    d_l_i: int
        Used for fine-tuning distances in some cases
    favourites_value_range: (int, int) tuple
        The range of values that the student's value for one of her favourite courses can take
    non_favourites_value_range: (int, int) tuple
        The range of values that the student's value for one of her non-favourite courses can take
    complements_decay: float
        The rate with which the students' bonus for getitng more courses from the same bundle decreases on expectation
    substitutes_decay: float
        The rate with which the students' penalty for getting more courses fromt he same substitute set decreases on expectation
    fixed_comp_values: Bool
        If true: The value for a course from a complement set is exaclty its expected value
    seed: int
        The seed to be used



    Returns:
    ----------------------------------
    optimal_schedule: list of 0-1s
        optimal_schedule[i] = 1 if course x_i is contained in the optimal bundle for that student, 0 else
    total_time: float
        The total time of the function call, in seconds. Only returns it if time_output = True.
    """

    random.seed(seed)
    np.random.seed(seed)

    student_type = np.random.choice(thematic_types, p = thematic_type_probabilities)  # generate the student type according to the given vector of types and type probabilities
#     print("Student type: ", student_type)
    n_i_pop = np.random.choice(favourite_courses_number, p = favourite_courses_probabilities)  # generate the number of favourite courses that that student will have
#     print("Number of favourite courses to draw: ", n_i_pop)
    number_of_centers = np.random.choice(center_numbers, p = center_number_probabilities)
    n_i_npop = np.random.choice(non_favourite_courses_number, p = non_favourite_courses_probabilities)
#     print("Number of non-favourite courses to draw: ", n_i_npop)
#     print("Number of centers to draw: ", number_of_centers)

    additive_prefs, substitutes, complements, _, _ = student_preference_simluator(courses = course_buckets, grid = course_grid, type_i = student_type, n_i_pop = n_i_pop,
            n_i_npop = n_i_npop, i_p_pop = favourite_pop_probability, i_p_npop = non_favourite_npop_probability, n_cntr = number_of_centers,
            i_p_pop_cntr = favourite_center_probability, d_s_i = substitute_range, d_c_i = complement_range, d_l_i = d_l_i,
            complements_expected_value = complements_expected_value, substitutes_expected_value = substitutes_expected_value,
            pop_low = favourites_value_range[0], pop_high = favourites_value_range[1], complements_decay= complements_decay, substitutes_decay= substitutes_decay,
                npop_low = non_favourites_value_range[0], npop_high = non_favourites_value_range[1], seed_i = seed, fixed_comp_values= fixed_comp_values)

    overload_penalty, timegap_penalty, free_days_marginal_values = time_pref_generator(seed = seed, disable_time_prefs= disable_time_prefs)
    return additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values


def student(bundle, additive_prefs, substitutes, complements, timetable, overload_penalty = -7,
            free_days_marginal_values = [22, 14, 6, 0, 0], credit_units = [1 for i in range(30)], make_monotone = True):
    """
    This function simply calls the calculate_bundle_value_by_hand function to avoid having to parse all the parameters each time.
    """
    def sub_lists(l):
        lists = [[]]
        for i in range(len(l) + 1):
            for j in range(i):
                lists.append(l[j: i])
        return lists

    if (not make_monotone):

        return calculate_bundle_value_by_hand(bundle, timetable, credit_units = credit_units, additive_preferences= additive_prefs,
            timegap_penalty= -5, complements_sets= complements, substitutes_sets= substitutes, overload_penalty= overload_penalty, ignore_timegaps= True,
                              free_days_marginal_values= free_days_marginal_values, verbose = False)
    else:
        one_positions = [i for i in range(len(bundle)) if bundle[i] == 1]  # get the position of all the ones
        subsets = sub_lists(one_positions)         # get the indices of all the elements
        max_subset_value = 0
        for subset in subsets:
            x = [0 for i in range(len(bundle))]
            for i in subset:
                x[i] = 1   # get the subset bundle in binary form

            subset_value = calculate_bundle_value_by_hand(x, timetable, credit_units = credit_units, additive_preferences= additive_prefs,
            timegap_penalty= 0, complements_sets= complements, substitutes_sets= substitutes, overload_penalty= overload_penalty, ignore_timegaps= True,
                              free_days_marginal_values= free_days_marginal_values, verbose = False)

            max_subset_value = max(max_subset_value, subset_value)
        return max_subset_value

# The MIP function for tabu search:


def solve_student(course_timetable, course_prices, credit_units, budget, cu_limit, additive_preferences,  complements_sets = [], substitutes_sets = [], max_bundle_size = 6,
                  overload_penalty = -15, timegap_penalty = -10, free_days_marginal_values = [40, 30, 10, 1, 1], already_taken = [], days = 5, timeslots = 12,
                  ignore_timegaps = False,  verbose = True, print_solution = False, time_output = False, outputFlag = False, print_time = False,
                  seats_available = None, forbidden_bundles = None):
    """
    A function that finds the true optimal legal bundle of courses for a student, given his true preferences

    Parameters:
    ----------------------------------
    course_timetable : 3-D Aarray of shape (days, timeslots, )
        course_timetable[d][t]: A list of all the courses with a lecture on timeslot t of day d
    course_prices: 1-D array of shape (n_courses)
        course_prices[i]: The price of the i-th course
    credit_untis: 1-D array of shape (n_courses)
        credit_units[i]: The credit units of the i-th course
    budget: float
        That student's budget
    additive_preferences: list of floats
        additive_preferences[i]: The student's additive value for the i-th course
    complements_sets: list of tuples
        each tuple has as its first element the list of items belonging in that set of complements, and as its second element the complement function psi' of that bundle,
        e.g. complements_sets[0] = ([2,5,7,17,35], [0, 0.2, 0.4, 0.6, 0.8])
    substitutes_sets: list of tuples
        each tuple has as its first element the list of items belonging in that set of substitutes, and as its second element the substitute function xi of that bundle,
        e.g. substitutes_sets[0] = ([2,5,7,17,35], [0, -0.2, -0.4, -0.6, -0.8 ])
    max_bundle_size: int
        The maximum number of courses that the student can take in a semster
    overload_penalty: float
        How much negative utility the student has for having overloaded days
    timegap_penalty: float
        How much negative utility the student loses for each timegap he has to spend at school
    free_days_marginal_values: list of floats
        free_days_marginal_values[i]: The exta utility for having an (i+1)-th free day, provided the student already has i free days.
    already_taken: list of ints
        The list of all the courses that that student has already taken
    days : int
        The number of school days (e.g. 5)
    timeslots : int
        The number of time slots per day (e.g. 12)
    ignore_timegaps: Boolean
        If true: The timegap part of the objective and constraints are removed from the objective, improves the time performance of the MIP, as those are very costly
    verbose: boolean
        If true: The function writes the corresponding MIP to the file StudentLP.lp
    print_solution: boolean
        If true: The function prints in an easy to read format all binary variables set to 1 for debug purposes
    time_output: boolean
        If true: The function instead of returning the optimal schedule returns the tuple (optimal schedule, total_time)
    seats_available: numpy array of shape (n_courses, ) or None
        if it is provided: Restrict the optimization problem only to those courses with >0 seats available.
    forbidden_bundles: list of lists of 0-1s
        The optimal solution cannot be any bundle in that set.

    Returns:
    ----------------------------------
    optimal_schedule: list of 0-1s
        optimal_schedule[i] = 1 if course x_i is contained in the optimal bundle for that student, 0 else
    total_time: float
        The total time of the function call, in seconds. Only returns it if time_output = True.
    """

    preprocess_start = timer()
    studentProblem = gp.Model("Student MIP")
#     print("--- NEW MIP ---")

    # --- Variable Declaration  ---

    course_variables = studentProblem.addVars([i for i in range(len(additive_preferences))], name="x", vtype = GRB.BINARY)
    # variable x_i = 1 if student takes the i-th course, 0 else

    complement_variables = []
    for c in range(len(complements_sets)):  # at every iteration, appends to the list complement_set_variables
        complement_set_variables = []        # the variables G_{c,j,t} corresponding to a specific set c
        for j in complements_sets[c][0]:
            complement_item_variables = [studentProblem.addVar(name=f'G{c}_{j}_{t}', vtype= GRB.BINARY)
              for t in range(1, min(len(complements_sets[c][0]), max_bundle_size) + 1)]  # \bar{\tau} = \min \{ max possible bundle for the student, size of the set c\}
            complement_set_variables.append(complement_item_variables)
        complement_variables.append(complement_set_variables)
    # variable G_{c,j,t} = 1 if the student takes course j and has exaclty t courses from the set c

    substitute_variables = []
    for s in range(len(substitutes_sets)):  # at every iteration, appends to the list complement_set_variables
        substitute_set_variables = []        # the variables J_{s,j,t} corresponding to a specific set s
        for j in substitutes_sets[s][0]:
            substitute_item_variables = [studentProblem.addVar(name=f'J{s}_{j}_{t}', vtype = GRB.BINARY)
              for t in range(1, min(len(substitutes_sets[s][0]), max_bundle_size) + 1)]
            substitute_set_variables.append(substitute_item_variables)
        substitute_variables.append(substitute_set_variables)
    # variable J_{s,j,t} = 1 if the stoudent takes course j and has exactly t courses from the set s

    if(not ignore_timegaps):
        timeslot_gap_variables = studentProblem.addVars([(i, j) for i in range(days) for j in range(timeslots)], name='t_', vtype= GRB.BINARY)
        # variable t_{i,j} = 1 if that timeslot is a "gap" for which he has to stay at school, 0 else

        before_variables = studentProblem.addVars([(i, j) for i in range(days) for j in range(timeslots)], name='b_', vtype= GRB.BINARY)
        # variable b_{i,j} = 1 if the student has a course before timeslot (i,j), 0 else

        after_variables = studentProblem.addVars([(i, j) for i in range(days) for j in range(timeslots)], name='a_', vtype= GRB.BINARY)
        # variable a_{i,j} = 1 if the student has a course after timeslot (i,j), 0 else

        timeslot_free_variables = studentProblem.addVars([(i, j) for i in range(days) for j in range(timeslots)], name='f_', vtype= GRB.BINARY)
        # variable f_{i,j} = 1 if the student has timeslot (i,j) free, 0 else

    weekday_free_variables = studentProblem.addVars(range(days), name= 'W_', vtype = GRB.BINARY)
    # W_i = 1 if the student has any lectures on the i-th day, 0 else

    weekday_f_variables = studentProblem.addVars(range(days), name= 'f_', vtype = GRB.BINARY)
    # f_i = 1 if the student's week has at least i+1 free days, 0 else

    load_variables = studentProblem.addVars(range(days), name= 'L_', vtype = GRB.INTEGER, lb = 0, ub = timeslots)
    # L_i: the load of the student (in hours) on the i-th day

    Overload_variable = studentProblem.addVar(name = 'O', vtype = GRB.CONTINUOUS, lb = 0, ub = timeslots)
    # O: a continuous variable indicating how "overloaded" the busiest day is, compared to the average one for that student

    # --- Objective Declaration ---

    # as a start, add the additive objective
    obj = gp.quicksum(course_variables[i] * additive_preferences[i] * credit_units[i] for i in range(len(course_prices)))

    # add the thematic map part of the objective
    for (i, (item_set, value_function)) in enumerate(complements_sets):
        for j in range(len(item_set)):
            base_value = additive_preferences[item_set[j]] * credit_units[item_set[j]]
            obj.add(gp.quicksum((base_value * value_function[k]) * complement_variables[i][j][k] for k in range(len(complement_variables[i][j]))))

    for (i, (item_set, value_function)) in enumerate(substitutes_sets):
        for j in range(len(item_set)):
            base_value = additive_preferences[item_set[j]] * credit_units[item_set[j]]
            obj.add(gp.quicksum((base_value * value_function[k]) * substitute_variables[i][j][k] for k in range(len(substitute_variables[i][j]))))

    # add the #free days part of the objective
    obj.add(gp.quicksum(free_days_marginal_values[i] * weekday_f_variables[i] for i in range(days)))

    # add the "overloaded day" part of the objective
    obj.add(Overload_variable * overload_penalty)

    # add the "as few timegaps as possible" part of the objectice
    if(not ignore_timegaps):
        obj.add(timegap_penalty * gp.quicksum(timeslot_gap_variables))

    studentProblem.setObjective(obj, GRB.MAXIMIZE)

    # ---  Feasibility Contraints ---

    # add the budget constraint
    budget_constraint = studentProblem.addConstr(gp.quicksum(course_variables[i] * course_prices[i] for i in range(len(course_prices))) <= budget, name = 'budget')

    # add the credit unit constraint
    cu_constraint = studentProblem.addConstr(gp.quicksum(course_variables[i] * credit_units[i] for i in range(len(credit_units))) <= cu_limit, name = 'creditUnits')

    # add the overlapping courses constraint:
    for day in course_timetable:
        for timeslot in day:
            studentProblem.addConstr(gp.quicksum(course_variables[i] for i in timeslot) <= 1, name='overlaps')
    # for any timeslot of any day, the student can only have one of the courses with a lecture on that timeslot

    # add the constraint that a student can't take the same course twice
    studentProblem.addConstrs((course_variables[i] == 0 for i in already_taken), name= 'alreadyTaken')

    # add the constraint that a student can only pick from the courses with seats available
    if(seats_available is not None):
        for i in range(seats_available.shape[0]):
            if (seats_available[i] == 0):
                studentProblem.addConstr(course_variables[i] == 0, name = 'no_seat_available')

    # add the constraint forbidding some bundles (for not querying the same set twice)
    if(forbidden_bundles is not None):
        for bundle in forbidden_bundles:
            studentProblem.addConstr(gp.quicksum(course_variables[i] * bundle[i] for i in range(len(bundle))) <= np.sum(bundle), name='alreadyQueried')

    # --- Constraints used to calculate the objective value --

    for i in range(len(weekday_free_variables)):
        courses_in_that_day = list(set(functools.reduce(operator.iconcat, course_timetable[i], [])))  # get the course ids of all courses in that day
        studentProblem.addConstr(weekday_free_variables[i] * min(timeslots, len(courses_in_that_day)) >= gp.quicksum(course_variables[j] for j in courses_in_that_day), name='weekdayFreeConstraints')
        # if at least one element on the RHS is 1 -> forces the RHS variable to 1

    studentProblem.addConstrs((weekday_f_variables[i] * (i + 1) + gp.quicksum(weekday_free_variables) <= days for i in range(len(weekday_free_variables))), name = 'weekdayIndicatorContraints')
    # the constraints for the f_i variables

    for i in range(len(load_variables)):
        courses_in_that_day = functools.reduce(operator.iconcat, course_timetable[i], [])
        hours_of_courses = Counter(courses_in_that_day)
        studentProblem.addConstr(gp.quicksum(course_variables[element] * hours_of_courses[element] for element in hours_of_courses) == load_variables[i], name = 'LoadConstraint')
        # the load constraints

    # add the overload inequality constraints
    studentProblem.addConstrs((Overload_variable * (len(load_variables) - 1) >= (len(load_variables) - 1) * load_variables[i] - gp.quicksum(load_variables[j] for j in range(len(load_variables)) if j != i) for i in range(len(load_variables))),
                              name = 'OverloadConstraint')

    if(not ignore_timegaps):
        # add all the "before timeslot_{i,j}" constraints
        for i in range(days):
            for j in range(timeslots):
                courses_before = list(set(functools.reduce(operator.iconcat, course_timetable[i][0:j], [])))
                studentProblem.addConstr(before_variables[(i, j)] * min(timeslots, len(courses_before)) >= gp.quicksum(course_variables[k] for k in courses_before), name = 'BeforeConstraints')

        # add all the "after timeslot_{i,j}" constraints
        for i in range(days):
            for j in range(timeslots):
                courses_after = list(set(functools.reduce(operator.iconcat, course_timetable[i][j+1: timeslots], [])))
                studentProblem.addConstr(after_variables[(i, j)] * min(timeslots, len(courses_before)) >= gp.quicksum(course_variables[k] for k in courses_after), name = 'AfterConstraints')

        # add all the "timeslot_{i,j} free" equality constraints
        for i in range(days):
            for j in range(timeslots):
                courses_in_that_timeslot = course_timetable[i][j]
                studentProblem.addConstr(timeslot_free_variables[(i, j)] + gp.quicksum(course_variables[k] for k in courses_in_that_timeslot) == 1, name = 'TimeslotFreeConstraint')

        # add the "timeslot_{i,j} is an actual gap" inequality constraints
        studentProblem.addConstrs((timeslot_gap_variables[(i, j)] >= before_variables[(i, j)] + after_variables[i, j] + timeslot_free_variables[(i, j)] - 2 for i in range(days) for j in range(timeslots)), name = 'TimeslotGapConstraint')

    # add the feasibility constraints for G_{c,j,t}
    for (i, (item_set, _)) in enumerate(complements_sets):
        for j in range(len(item_set)):
            studentProblem.addConstr(gp.quicksum(complement_variables[i][j][k] for k in range(len(complement_variables[i][j]))) <= course_variables[item_set[j]], name = 'CompFeasibilityContraint')

    # add the \tau constraints for G_{c,j,t}
    for (i, (item_set, _)) in enumerate(complements_sets):
        for j in range(len(item_set)):
            studentProblem.addConstr(gp.quicksum(complement_variables[i][l][k] for l in range(len(complement_variables[i]))
                                        for k in range(len(complement_variables[i][l]))) >= gp.quicksum((k+1) * complement_variables[i][j][k] for k in range(len(complement_variables[i][j]))), name = 'CompTauConstraint')

    # add the feasibility constraints for J_{s,j,t}
    for (i, (item_set, _)) in enumerate(substitutes_sets):
        for j in range(len(item_set)):
            studentProblem.addConstr(gp.quicksum(substitute_variables[i][j][k] for k in range(len(substitute_variables[i][j]))) >= course_variables[item_set[j]], name = 'SubFeasibilityConstr')
            studentProblem.addConstr(gp.quicksum(substitute_variables[i][j][k] for k in range(len(substitute_variables[i][j]))) <= 1, name ='SubFeasibilityConstr2')

    # add the \tau constraints for J_{s,j,t}
    for (i, (item_set, _)) in enumerate(substitutes_sets):
        for j in range(len(item_set)):
            studentProblem.addConstr(gp.quicksum(substitute_variables[i][l][k] for l in range(len(substitute_variables[i])) for
                                                  k in range(len(substitute_variables[i][l]))) <= len(substitute_variables[i][j]) + gp.quicksum((k+1 - len(substitute_variables[i][j])) * substitute_variables[i][j][k] for
                                                                                                                                                   k in range(len(substitute_variables[i][j]))), name = 'SubTauConstraint')

    # --- Calculate optimal Solution ---
    start = timer()
    if(verbose):
        studentProblem.write('StudentLP.lp')
    studentProblem.Params.OutputFlag = outputFlag
    studentProblem.optimize()
    end = timer()

    total_time = end - preprocess_start

    if(print_time):
        print("Time for preprocessing: ", start - preprocess_start)
        print("Time elapsed for MIP optimization: ", end - start)
        print("Total time: ", total_time)

    optimal_schedule = []

    for i in range(len(course_variables)):
        if(course_variables[i].x >= 0.99):
            optimal_schedule.append(1)
        else:
            optimal_schedule.append(0)

    # print the soulution in an easy to read format for debug purposes
    if(print_solution):
        courses_picked_strings = []
        for i in range(len(course_variables)):
            if(course_variables[i].x >= 0.99):
                courses_picked_strings.append(course_variables[i].VarName)
        print("Courses in the optimal solution:", courses_picked_strings)

        complement_vars_picked_strings = []
        for i in range(len(complement_variables)):
            for j in range(len(complement_variables[i])):
                for k in range(len(complement_variables[i][j])):
                    if(complement_variables[i][j][k].x >= 0.999):
                        complement_vars_picked_strings.append(complement_variables[i][j][k].VarName)
        print("Complement vars in the optimal solution:", complement_vars_picked_strings)

        substitute_vars_picked_strings = []
        for i in range(len(substitute_variables)):
            for j in range(len(substitute_variables[i])):
                for k in range(len(substitute_variables[i][j])):
                    if(substitute_variables[i][j][k].x >= 0.999):
                        substitute_vars_picked_strings.append(substitute_variables[i][j][k].VarName)
        print("Substitute vars in the optimal solution:", substitute_vars_picked_strings)

    if (time_output):
        return (optimal_schedule, total_time, studentProblem.getObjective().getValue())

    return optimal_schedule
