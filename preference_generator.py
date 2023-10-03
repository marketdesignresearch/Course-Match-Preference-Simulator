import numpy as np 
import math 
import random

from preference_generator_utils import courses, make_grid, generate_student_preferences, save_obj, solve_student, calculate_bundle_value_by_hand
from pdb import set_trace



def create_multiple_students(number_of_students = 100, number_of_courses = 30,  number_of_popular = 8, mean_number_of_favourites = 4.87, number_of_centers = 2, gr_height = 5, gr_width = 6, seed = 42,
                            base_budget = 1, complement_range = 1, substitute_range = 1, complements_expected_value = 0.4,
                             substitutes_expected_value = 0.4, favourites_value_range = (80, 120), non_favourites_value_range = (40, 60), complements_decay = 0.8, substitutes_decay = 0.8,
                             disable_time_prefs = True, fixed_comp_values = True, maximum_budget_deviation_beta = 0.01):
    """
    A function generating a complete student list for a run of Course Match, including the student preferences and their budgets.

    Parameters:
    --------------
    number_of_students: int
        The number of students
    number_of_courses: int
        The number of courses
    number_of_popular: int
        The number of popular courses
    nubmer_of_favourites: int
        The number of favourite courses. They will be drawn exclusively from the popular courses!
    number_of_centers: int
        The number of complement/substitute centers. They will be drawn exclusively from the student's favourite courses!
    gr_height: int
        The height of the grid of the thematic map
    gr_width: int
        The width of the grid of the thempatic map
    seed: int
        The random seeed to set
    base_budget: float
        The mean budget of a student
    budget_spread: float
        The spread of the students' budgets
    complement_range: int
        Defines the range within which the courses are complements to the center
    substitute_range: int
        Defines the range within which the courses are substitutes to the center
    complements_expected_value: float
        The expected bonus for getting more courses from the same set of complements
    substitutes_expected_value: float
        The expected penalty for getting more courses from the same set of substitutes
    favourites_value_range: (int, int) tuple
        The value of a favourite course for each student is drawn uniformly from that range
    non_favourites_value_range: (int, int) tuple
        The value of a favourite course for each student is drawn uniformly from that range
    disable_time_prefs: Bool
        If true: Students have no time preferences.
    fixed_comp_values: Bool
        If true: The students complement bonus/substitute penalty is always equal to its expected value.


    """
    if(seed is not None):  # just so that we don't by mistake set a random random seed
        random.seed(seed)
        np.random.seed(seed)

    course_buckets = courses(n_types = 1, n_courses= number_of_courses,
                n_A_pop = number_of_popular, n_A_npop = number_of_courses - number_of_popular, seed = seed)  # create the course buckets (same for all students)
    grid = make_grid(n_courses = number_of_courses, height = gr_height, width = gr_width, seed = seed)   # create the course grid (same for all students)
    budget_step = maximum_budget_deviation_beta / number_of_students

    student_list = []
    for i in range(number_of_students):
        random_number = np.random.uniform(low = 0, high = 1)
        if random_number <= mean_number_of_favourites - math.floor(mean_number_of_favourites):
            number_of_favourites = int(math.ceil(mean_number_of_favourites))
        else:
            number_of_favourites = int(math.floor(mean_number_of_favourites))

        (additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values) = generate_student_preferences(course_buckets,
                        grid, complement_range = complement_range, substitute_range = substitute_range, favourite_courses_number= [number_of_favourites],
                        non_favourite_courses_number= [number_of_courses - number_of_favourites], non_favourite_courses_probabilities= [1],
                        complements_expected_value = complements_expected_value, substitutes_expected_value = substitutes_expected_value,
                        seed = seed + i, disable_time_prefs= disable_time_prefs, favourites_value_range= favourites_value_range, center_numbers= [number_of_centers], center_number_probabilities= [1],
                        non_favourites_value_range = non_favourites_value_range, complements_decay= complements_decay, substitutes_decay= substitutes_decay, fixed_comp_values= fixed_comp_values)

        budget = base_budget + i * budget_step

        student_list.append((additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, budget))

    return student_list

def capacities_generator(number_of_courses = 30, total_number_of_seats = 505, capacity_deviation = 0, seed = 42):
    """
    A random function generating the capacity of each course, given the total capacity:

    Parameters:
    -----------------
    number_of_courses: int
        The number of courses available.
    total_number_of_seats: int
        The total number of seats of all courses.
    capacity_deviation: float
        The standard deviation of the capacities' uniform distribution. Do not go larger than!
    seed: int
        The random seed used

    Returns:
    -----------------
    capacities: numpy array of shape (number_of_courses, )
        capacities[i]: The capacity of the i-th course
    """
    rng = np.random.default_rng(seed = seed)
    values = rng.uniform(low = 1 - capacity_deviation, high = 1, size = number_of_courses)
    capacities_float = (values / values.sum()) * total_number_of_seats
    capacities = np.rint(capacities_float)

    return capacities

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


def generate_problem_instance_principled(number_of_times = 10, number_of_courses = 30, number_of_students = 100, supply_ratio = 1.25, capacity_deviation = 0, grid_heigth = 5, grid_width = 6, complement_range = 1, substitute_range = 1,
        complements_expected_value = 0.4, substitutes_expected_value = 0.4, complements_decay = 0.8, substitutes_decay = 0.8, favourites_value_range = (80, 120), non_favourites_value_range = (40, 60),
        number_of_popular = 8, mean_number_of_favourites = 2, number_of_centers = 2, disable_time_prefs = True, fixed_comp_values = True, maximum_budget_deviation_beta = 0.01, seed = 42):
    """
    Generates true student profiles for the number of runs specified so that they can then be fed in the different individual stages of course match.
    """

    true_student_list_all_runs = []
    capacities_all_runs = []
    timetables_all_runs = []

    for i in range(number_of_times):
        print(f'Generating students for run {i}')
        np.random.seed(seed + (i * number_of_students))
        student_list = create_multiple_students(number_of_students, number_of_courses, gr_height= grid_heigth, gr_width= grid_width,
            complement_range= complement_range, substitute_range= substitute_range, complements_expected_value = complements_expected_value, substitutes_expected_value = substitutes_expected_value,
            seed = seed + (i * number_of_students), favourites_value_range= favourites_value_range, non_favourites_value_range= non_favourites_value_range,
            number_of_popular= number_of_popular, mean_number_of_favourites = mean_number_of_favourites, number_of_centers = number_of_centers,
            disable_time_prefs = disable_time_prefs, fixed_comp_values= fixed_comp_values, maximum_budget_deviation_beta= maximum_budget_deviation_beta)
        capacities = capacities_generator(number_of_courses = number_of_courses, total_number_of_seats= number_of_students * 5 * supply_ratio, capacity_deviation= capacity_deviation, seed = seed + i)
        timetable = timetable_generator(number_of_courses, credit_units= [0.5 for i in range(int(math.floor(number_of_courses / 2)))] + [1 for i in range(int(math.ceil(number_of_courses / 2)))], seed = seed + i)

        true_student_list_all_runs.append(student_list)
        capacities_all_runs.append(capacities)
        timetables_all_runs.append(timetable)

    return true_student_list_all_runs, np.array(capacities_all_runs), timetables_all_runs


def create_all_instances(number_of_instances = 100, number_of_courses = 25, supply_ratio = 1.25, number_of_popular = 9, mean_number_of_favourites = 4.87, seed = 8267, large_grid = True, 
                         maximum_budget_deviation_beta = 0.04, additive_preferences = False, save_results = False, save_folder = 'problem_instances_final'):
    """
    A  top-level function that creates and saves instances of the true student preferences according to the parameters specified in the paper.

    Parameters:
    ----------------------------------
    number_of_instances : int
        The number of instances to be created
    number_of_courses : int
        The number of courses avaiable. 
    supply_ratio : float
        The school's supply ratio, defined as the fraction of total seat demand by the students to the total number of seats available.
    number_of_popular: int 
        The number of popular courses in the school, defined as in Section 5 of our paper. The smaller this number is, the more correlated the students' preferences will be.
    mean_number_of_favourites: float
        The number of courses for which a studnet will on average have a high value for. 
    seed: int 
        The seed to be used for the random number generator.
    large_grid: bool
        Affects the size of the grid determinign complementarities and substitutabilities, as described in section 5 of our paper. A smaller grid will result in students having more complementarities and substitutabilities.
    maximum_budget_deviation_beta: float
        The maximum budget deviation beta between the lowest and highest budget of a student. 
    additive_preferences: bool
        If true, the students' preferences will be additive. If false, the students will have complements/substitutes
    save_results: bool
        Whether to save the results to a file or not. 
    save_folder: str
        The folder name to save the results to.


    Returns:
    ----------------------------------
    true_student_lists_all_instances: list of negth <number of instances>
        true_student_lists_all_instances[i][j]: The j-th student's preferences in the i-th instance
    capacities_all_instances: numpy array of shape (<number of instances>, <number of courses>)
        capacities_all_instances[i][j]: The capacity of the j-th course in the i-th instance
    timetables_all_instances: list of length <number of instances>
        timetables_all_instances[i]: List of nength <number of school days> x <number of time slots per day>.
        timetables_all_instances[i][j][k]: List of all courses being taught in the k-th time slot of the j-th day of the i-th instance   
    """
     
    if large_grid: 
        grid_h = 7
        grid_w = 7
        
    else: 
        grid_h = 5
        grid_w = 6

    if additive_preferences:
        complement_range = 0
        substitute_range = 0
    else:
        complement_range = 1
        substitute_range = 1
        
    true_student_lists_all_instances, capacities_all_instances, timetables_all_instances = generate_problem_instance_principled(number_of_times= number_of_instances,
        number_of_courses= number_of_courses,
        number_of_students=100,
        supply_ratio= supply_ratio,
        capacity_deviation=0,
        grid_heigth=grid_h,           
        grid_width=grid_w,     
        complement_range=complement_range,
        substitute_range=substitute_range,
        complements_expected_value=0.4,
        substitutes_expected_value=0.4,
        complements_decay=0.8,
        substitutes_decay=0.8,
        favourites_value_range=(50, 100),       
        non_favourites_value_range=(0, 50),    
        number_of_popular=number_of_popular,    
        mean_number_of_favourites= mean_number_of_favourites,       
        number_of_centers=2,                   
        disable_time_prefs=True,
        fixed_comp_values=True,
        maximum_budget_deviation_beta= maximum_budget_deviation_beta,
        seed= seed)


    if save_results:
        save_obj(true_student_lists_all_instances, f'./{save_folder}/true_student_lists_sr_{supply_ratio}_popular_{number_of_popular}_lg_{large_grid}_additive_preferences_{additive_preferences}')
        save_obj(timetables_all_instances, f'./{save_folder}/timetables_sr_{supply_ratio}_popular_{number_of_popular}_lg_{large_grid}_additive_preferences_{additive_preferences}')
        np.save(f'./{save_folder}/capacities_all_runs_sr_{supply_ratio}_popular_{number_of_popular}_lg_{large_grid}_additive_preferences_{additive_preferences}', capacities_all_instances, allow_pickle= False)
    
    return true_student_lists_all_instances, capacities_all_instances, timetables_all_instances


def noisify_student(individual_student, forget_base = 0.5, forget_base_uniform= 0, forget_adjustments = 0.2, base_noise_std = 10, adjustment_noise_std = 0.2, seed = 42, return_forgotten_bases = False, multiplicative_base_noise = False):
    """
    A function taking as input the true preferences of a single student and reutrning a "noisy" version of that. 

    Paramters:
    -----------------
    inidividual_student: tuple
        The true preferences of a single student, in the format of the true_student_list
    forget_base: float
        The percentage of her lowest base values that a student will forget.
    forget_base_uniform: float 
        An additional probability that she forgets a base value, regardless of the value of the course
    forget_adjustments: float
        The percentage of her adjustments that a student will forget.
    base_noise_std: float
        The std of the noise that will be added to the students' base values.
    adjustment_noise_std: float
        The std of the noise that will be added to the students' adjustments.
    seed: int
        The random seed to be used

    Returns:
    -----------------
    noisy_student: tuple
        A student of the same type as the one returned by create_multiple_student, but with the appropriate noise added to it.
        if return_forgotten_bases: Returns base values, complements, substitutes and additionally for which courses the base values were forgotten
    """

    # print(f'Noisify student got called with forget_base_uniform: {forget_base_uniform}')
    base_values_copy = individual_student[0].copy()


    # add noise to the base values of the courses
    rng = np.random.default_rng(seed)
    if not multiplicative_base_noise:
        base_noise = rng.normal(scale = base_noise_std, size = base_values_copy.shape)
        base_values_copy = base_values_copy + base_noise

    else:
        print('noisify student called with multiplicative_base_noise!!!')
        base_noise = rng.uniform(low = 1 - base_noise_std, high = 1 + base_noise_std, size = base_values_copy.shape)
        base_values_copy = base_values_copy * base_noise
        # print(f'base values noise: {base_noise}')

    base_values_copy = np.maximum(base_values_copy, 0)  # make sure the base values do not go below 0!
#     set_trace()

    # forget some of the courses
    sorted_arguments = np.argsort(base_values_copy)

    expected_number_of_courses_to_forget = base_values_copy.shape[0] * forget_base
    random_number = rng.uniform(low = 0, high = 1)
    if random_number <= expected_number_of_courses_to_forget - math.floor(expected_number_of_courses_to_forget):
        courses_to_forget = math.ceil(expected_number_of_courses_to_forget)
    else:
        courses_to_forget = math.floor(expected_number_of_courses_to_forget)

    unforgotten_bases = np.array([1 for i in range(len(base_values_copy))])

    base_values_copy[sorted_arguments[:courses_to_forget]] = 0
    unforgotten_bases[sorted_arguments[:courses_to_forget]] = 0  # also mark the courses you forgot to report a base value for.

    for i in range(len(base_values_copy)):
        if (rng.random() < forget_base_uniform):
            base_values_copy[i] = 0
            unforgotten_bases[i] = 0
            print(f'Forgetting course {i} because of the new condition')

#     set_trace()

    # forget some of the adjustmnets
    noisy_substitutes = []
    noisy_complements = []

    for (substitute_list, substitute_values) in individual_student[1]:
        substitute_list_clipped = [x for x in substitute_list if base_values_copy[x] != 0]
        unforgettable_substitutes = [x for x in substitute_list_clipped if rng.random() >= forget_adjustments]
        if(len(unforgettable_substitutes) > 1):
            unforgettable_values = substitute_values[: len(unforgettable_substitutes)]
            # adjustment_noise = rng.normal(loc = 1.0,  scale = adjustment_noise_std, size = unforgettable_values.shape)
            adjustment_noise = rng.uniform(low = 1 - adjustment_noise_std, high = 1 + adjustment_noise_std, size = unforgettable_values.shape)
            unforgettable_values = unforgettable_values * adjustment_noise
            unforgettable_values = np.minimum(unforgettable_values, 0)  # make sure substitutes don't become complements!
            unforgettable_values[::-1].sort()  # make sure they are still sorted properly!
            noisy_substitutes.append((unforgettable_substitutes, unforgettable_values))

    for (complement_list, complement_values) in individual_student[2]:
        complement_list_clipped = [x for x in complement_list if base_values_copy[x] != 0]
        unforgettable_complements = [x for x in complement_list_clipped if rng.random() >= forget_adjustments]
        if(len(unforgettable_complements) > 1):
            unforgettable_values = complement_values[: len(unforgettable_complements)]
            # adjustment_noise = rng.normal(loc = 1.0,  scale = adjustment_noise_std, size = unforgettable_values.shape)
            adjustment_noise = rng.uniform(low = 1 - adjustment_noise_std, high = 1 + adjustment_noise_std, size = unforgettable_values.shape)
            unforgettable_values = unforgettable_values * adjustment_noise
            unforgettable_values = np.maximum(unforgettable_values, 0)   # make sure complements don't become substitutes!
            unforgettable_values.sort()                                 # make sure they are still sorted properly!
            noisy_complements.append((unforgettable_complements, unforgettable_values))

    if not return_forgotten_bases:
        return (base_values_copy, noisy_substitutes, noisy_complements, individual_student[3], individual_student[4], individual_student[5], individual_student[6])

    else:

        return (base_values_copy, noisy_substitutes, noisy_complements, unforgotten_bases)  # if return_forgotten_bases: only returns base values/complements/substitutes and what the student reported no value for


def keep_pairwise_adjustments(complements, substitutes):
    """
    A fucntion taking as input the true complements and substitutes of a student (i.e., bundles of all possible size), and returning only the pairwise adjustments.
    """
    complements_clipped = []
    substitutes_clipped = []
    for (course_indexes, adjustments) in complements:
        adjustments_clipped = np.full(adjustments.shape, adjustments[1])
        adjustments_clipped[0] = 0
        complements_clipped.append((course_indexes, adjustments_clipped))

    for (course_indexes, adjustments) in substitutes:
        adjustments_clipped = np.full(adjustments.shape, adjustments[1])
        adjustments_clipped[0] = 0
        substitutes_clipped.append((course_indexes, adjustments_clipped))

    return(complements_clipped, substitutes_clipped)


def noisify_all_students(student_list, forget_base = 0.15, forget_base_uniform= 0, forget_adjustments = 0.0, base_noise_std = 3, adjustment_noise_std = 0.05, seed = 42, multiplicative_base_noise = False):
    """
    A function taking as input the true preferences of all students and reutrning a "noisy" version of that, much more efficiently than noisifying bundles one by one.

    Paramters:
    -----------------
    student_list: list
        A list containing the true preferences of all students, as returned by create_multiple_students
    forget_base: float
        The percentage of her lowest base values that a student will forget.
    forget_adjustments: float
        The percentage of her adjustments that a student will forget.
    base_noise_std: float
        The std of the noise that will be added to the students' base values.
    adjustment_noise_std: float
        The std of the noise that will be added to the students' adjustments.
    seed: int
        The random seed to be used

    Returns:
    -----------------
    noisy_list: list
        A list of the same type as the one returned by create_multiple_students, but with the appropriate noise added to every student.
    """

    noisy_list = []
    print(f'Noisy all students got called with forget base uniform: {forget_base_uniform}')
    for (i, individual_student) in enumerate(student_list):
        noisy_student = noisify_student(individual_student, forget_base= forget_base, forget_base_uniform= forget_base_uniform, forget_adjustments = forget_adjustments,
                                        base_noise_std= base_noise_std, adjustment_noise_std = adjustment_noise_std, seed = seed + i, multiplicative_base_noise= multiplicative_base_noise)
        noisy_list.append(noisy_student)

    return noisy_list


def create_noisy_model_student_list(student_list, model_type, seed, model_param_dictionary = None):
    """
    The heart of the new Course Match. A function taking as input the true student_list and creating the corresponding one for any model/algorithm.

    Paramters:
    -----------------
    student_list: list
        The list of the true students' preferences, as returned by create_multiple_students
    model_type: string
        The model for which to create a student list.
    seed: int
        The random seeed to be used.
    model_param_dictionary: dictionary or None
        Depending on the model, contains all the remaining model parameters required.
    """
    if (model_type == 'True'):
        return student_list

    elif (model_type == 'TrueNoisy'):
        return noisify_all_students(student_list, forget_base = model_param_dictionary['noisy_forget_base'], forget_base_uniform= model_param_dictionary.get('noisy_forget_base_uniform', 0),
                forget_adjustments = model_param_dictionary['noisy_forget_adjustments'],
                base_noise_std = model_param_dictionary['noisy_base_std'],  adjustment_noise_std= model_param_dictionary['noisy_adj_std'], seed = seed,
                multiplicative_base_noise= model_param_dictionary.get('multiplicative_base_noise', False))

    elif (model_type == 'TrueLinear'):
        linear_student_list = []
        for (additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, budget) in student_list:
            linear_student_list.append((additive_prefs, budget))
        return linear_student_list

    elif (model_type == 'LinearNoisy'):
        noisy_student_list = noisify_all_students(student_list, forget_base = model_param_dictionary['noisy_forget_base'], forget_base_uniform= model_param_dictionary.get('noisy_forget_base_uniform', 0),
                forget_adjustments = model_param_dictionary['noisy_forget_adjustments'],
                base_noise_std = model_param_dictionary['noisy_base_std'],  adjustment_noise_std= model_param_dictionary['noisy_adj_std'], seed= seed,
                multiplicative_base_noise= model_param_dictionary.get('multiplicative_base_noise', False))
        linear_student_list = []
        for (additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, budget) in noisy_student_list:
            linear_student_list.append((additive_prefs, budget))
        return linear_student_list

    elif(model_type == 'PairwiseAdjustments'):
        PA_student_list = []
        for (additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, budget) in student_list:
            complements_clipped, substitutes_clipped = keep_pairwise_adjustments(complements, substitutes)
            PA_student_list.append((additive_prefs, substitutes_clipped, complements_clipped, budget))
        return PA_student_list

    elif(model_type == 'PairwiseAdjustmentsNoisy'):
        noisy_student_list = noisify_all_students(student_list, forget_base = model_param_dictionary['noisy_forget_base'], forget_base_uniform= model_param_dictionary.get('noisy_forget_base_uniform', 0),
                forget_adjustments = model_param_dictionary['noisy_forget_adjustments'],
                base_noise_std = model_param_dictionary['noisy_base_std'],  adjustment_noise_std= model_param_dictionary['noisy_adj_std'], seed= seed,
                multiplicative_base_noise= model_param_dictionary.get('multiplicative_base_noise', False))
        PA_student_list = []
        for (additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, budget) in noisy_student_list:
            complements_clipped, substitutes_clipped = keep_pairwise_adjustments(complements, substitutes)
            PA_student_list.append((additive_prefs, substitutes_clipped, complements_clipped, budget))
        return PA_student_list

    raise ValueError(f'Unknown model type: {model_type}')


def calculate_single_student_demand(prices, student_profile, course_timetable, credit_units = [1 for i in range(25)], model_type = 'True', credit_units_per_student = 5, budget = None):
    """
    Takes as input a price vector of courses a students' preferences and optionally her budget and returns the optimal legal schedule for that student. 
    If no budget is given: Her initial budget will be used. 

    Parameters:
    --------------------
    prices: np.array of shape(number_of_units, )
        prices[i]: The price of the i-th course
    student_profile: The prefences of that student, in the form of either her true preferences or her GUI reports
    course_timetable: list of lists of ints
        course_timetable[i][j]: The ids of all courses being taught in the j-th timeslot of the i-th day
    credit_units: list of floats
        credit_units[i]: The credit units of the i-th course
    model_type: string
        The type of problem instance to solve. e.g. for `True' it solves the MIP of the true preferences of each student.
    courses_per_student: int
        The maximum nubmer of courses each student is willing to take.

    Returns:
    student_demand: np.array of shape (number_of_courses, )
        sutdnet_demand[i]: 1 if the i-th course is included in the student's optimal schedule, 0 otherwise
    """
    total_demand = np.zeros(prices.shape[0])

    individual_demands = []

    if (model_type == 'True' or model_type == 'TrueNoisy'):
        (additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, initial_budget) = student_profile
        if budget is None:
            budget = initial_budget

        student_demand = solve_student(course_timetable, prices, credit_units, budget, credit_units_per_student, additive_prefs, complements, substitutes, overload_penalty = overload_penalty,
                        timegap_penalty= timegap_penalty, free_days_marginal_values= free_days_marginal_values, ignore_timegaps= True, verbose = False)

    elif(model_type == 'TrueLinear' or model_type == 'LinearNoisy'):
        (linear_coefficients, initial_budget) = student_profile
        if budget is None:
            budget = initial_budget

        student_demand = solve_student(course_timetable, prices, credit_units, budget, credit_units_per_student, linear_coefficients, [], [], overload_penalty = 0,
                        timegap_penalty= 0, free_days_marginal_values= [0, 0, 0, 0, 0], ignore_timegaps= True, verbose = False)


    elif(model_type == 'PairwiseAdjustments' or model_type == 'PairwiseAdjustmentsNoisy'):
        (additive_prefs, substitutes_clipped, complements_clipped, initial_budget) =  student_profile
        if budget is None:
            budget = initial_budget

        student_demand = solve_student(course_timetable, prices, credit_units, budget, credit_units_per_student, additive_prefs, complements_clipped, substitutes_clipped,
                                        overload_penalty = 0, timegap_penalty= 0, free_days_marginal_values= [0, 0, 0, 0, 0], ignore_timegaps= True, verbose = False)


    return np.array(student_demand)

def calculate_total_demand(prices, student_profiles, course_timetable, credit_units = [1 for i in range(30)], return_individual_demands = False,
                           model_type = 'True', model_param_dictionary = None, credit_units_per_student = 5):
    """
    Takes as input a price vector of courses and the students' preferences and returns the total demand for each course, and optionally the individual demand of each student

    Parameters:
    --------------------
    prices: np.array of shape(number_of_units, )
        prices[i]: The price of the i-th course
    student_profiles: list of tuples of the form (additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, budget)
        student_profile[i]: Those numbers for the i-th student
    course_timetable: list of lists of ints
        course_timetable[i][j]: The ids of all courses being taught in the j-th timeslot of the i-th day
    credit_units: list of floats
        credit_units[i]: The credit units of the i-th course
    model_type: string
        The type of problem instance to solve. e.g. for `True' it solves the MIP of the true preferences of each student.
    courses_per_student: int
        The maximum nubmer of courses each student is willing to take.

    Returns:
    total_demand: np.array of shape (number_of_courses, )
        total_demand[i]: The total demand of all students for the i-th course
    """
    total_demand = np.zeros(prices.shape[0])

    individual_demands = []

    if (model_type == 'True' or model_type == 'TrueNoisy'):
        for (additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, budget) in student_profiles:
            student_demand = solve_student(course_timetable, prices, credit_units, budget, credit_units_per_student, additive_prefs, complements, substitutes, overload_penalty = overload_penalty,
                            timegap_penalty= timegap_penalty, free_days_marginal_values= free_days_marginal_values, ignore_timegaps= True, verbose = False)

            total_demand = total_demand + np.array(student_demand)
            individual_demands.append(np.array(student_demand))

    elif(model_type == 'TrueLinear' or model_type == 'LinearNoisy'):
        for (linear_coefficients, budget) in student_profiles:
            student_demand = solve_student(course_timetable, prices, credit_units, budget, credit_units_per_student, linear_coefficients, [], [], overload_penalty = 0,
                            timegap_penalty= 0, free_days_marginal_values= [0, 0, 0, 0, 0], ignore_timegaps= True, verbose = False)

            total_demand = total_demand + np.array(student_demand)
            individual_demands.append(np.array(student_demand))

    elif(model_type == 'PairwiseAdjustments' or model_type == 'PairwiseAdjustmentsNoisy'):
        for (additive_prefs, substitutes_clipped, complements_clipped, budget) in student_profiles:
            student_demand = solve_student(course_timetable, prices, credit_units, budget, credit_units_per_student, additive_prefs, complements_clipped, substitutes_clipped,
                                           overload_penalty = 0, timegap_penalty= 0, free_days_marginal_values= [0, 0, 0, 0, 0], ignore_timegaps= True, verbose = False)

            total_demand = total_demand + np.array(student_demand)
            individual_demands.append(np.array(student_demand))

    else:
        raise ValueError(f'Unknown model type: {model_type}')

    if(return_individual_demands):
        return (total_demand, np.array(individual_demands))

    return total_demand


def calculate_true_bundle_value(bundle, student_preferences, timetable, make_monotone = True):
    """
    This function takes as input the true preferences of a student and a bundle and outputs the true value of the student for that bundle 

    Parameters:
    --------------------
    bundle: np.array of shape(number of courses, )
        bundle[i]: 1 if the i-th course is contained in the bundle, 0 otherwise
        This is the format returned by calculate_total_demand
    student_preferences: tuple of the form (additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, budget)
        This is the format in which the true student's preferences are stored.
    course_timetable: list of lists of ints
        course_timetable[i][j]: The ids of all courses being taught in the j-th timeslot of the i-th day
    make_monotone: bool 
        If monotonicity should be enforced by calculate total value (default true). 
        The motivation is that a period where students can drop courses exists, so the value of a student for a schedule of courses is equal to the 
        student's value for the highest subset of courses contained in that bundle. 

    Returns: float
    The value of the student for the given course bundle. 
    """
    def sub_lists(l):
        lists = [[]]
        for i in range(len(l) + 1):
            for j in range(i):
                lists.append(l[j: i])
        return lists
    
    additive_prefs, substitutes, complements, overload_penalty, timegap_penalty, free_days_marginal_values, budget = student_preferences
    credit_units = [1 for i in range(len(additive_prefs))]

    if (not make_monotone):

        return calculate_bundle_value_by_hand(bundle, timetable, credit_units = credit_units, additive_preferences= additive_prefs,
            timegap_penalty= timegap_penalty, complements_sets= complements, substitutes_sets= substitutes, overload_penalty= overload_penalty, ignore_timegaps= True,
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
            timegap_penalty=  timegap_penalty, complements_sets= complements, substitutes_sets= substitutes, overload_penalty= overload_penalty, ignore_timegaps= True,
                              free_days_marginal_values= free_days_marginal_values, verbose = False)

            max_subset_value = max(max_subset_value, subset_value)
        return max_subset_value

    
    
    