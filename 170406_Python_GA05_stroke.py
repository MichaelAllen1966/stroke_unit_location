import numpy as np
import pandas as pd
import random as rn
import pyf_ga05_functions_170406 as ga
import datetime
from scipy.spatial.distance import pdist
import os

print('start', datetime.datetime.now().strftime("%H:%M:%S"))

# Set output location create folder if one does not already exist
OUTPUT_LOCATION = 'output/stroke/170206_600adm_3_10'
if not os.path.exists(OUTPUT_LOCATION):
    os.makedirs(OUTPUT_LOCATION)

# Directories needed
# Data required to run algorith to be stored in 'data' folder as subfolder of where code is stored


##Defining scores and which are used in Pareto selection
# Score_matrix:
# 0: Number of hospitals
# 1: Average distance
# 2: Maximum distance
# 3: Maximum admissions to any one hopsital
# 4: Minimum admissions to any one hopsital
# 5: Max/Min Admissions ratio
# 6: Proportion patients within target time/distance 1
# 7: Proportion patients within target time/distance 2
# 8: Proportion patients within target time/distance 3
# 9: Proportion patients attending unit with target admission numbers
# 10: Proportion of patients meeting time/distance 1 and admissions target
# 11: Proportion of patients meeting time/distance 2 and admissions target
# 12: Proportion of patients meeting time/distance 3 and admissions target
# 13: Clinical benefit if thrombolysis (fixed door to needle = mins + fixed onset to travelling in ambulance time = mins + travel time which is model dependant).  Additional benefit per 100 treatable patients

pareto_include = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])  # scores to use in defining pareto front
nscore_parameters = 14 # total parameter count to calculate, leave at 14
rn.seed()  # selects random number seed based on clock, Use fixed value in argument for fixed random number sequences

## Import data
# A starting population may be imported
# This is in the form of a csv with binary 0/1 for hospitals (in each column) closed or open
LOAD_INITAL_POPULATION = 0  # Set to 1 to load an initial population. This will be added to any randomly generated population
if LOAD_INITAL_POPULATION == 1:
    LOAD_POPULATION = np.loadtxt('data/data_filename.csv', delimiter=',')
HOSPITALS = pd.read_csv('data/hospitals.csv')  # Read hospital info into dataframe.
HOSPITAL_COUNT = len(HOSPITALS.index) # Count number fo hospitals

# Fixing hospitals as forced open or forced closed
# Column 9 of 'hospitals' may contain value of -1 to force a hopsital always to be closed, 1 to open, 0 to vary

DO_FIX_HOSPITAL_STATUS = 0  # Use 0 to ignore fixed hospital open/closed list, 1 to use list
a = np.array(HOSPITALS)  # Convert panda to np.array
HOSPITAL_STATUS = np.array(a[:, 9]).reshape(
    (1, HOSPITAL_COUNT))  # take the 5th column from the file and make it a single row
del a # remove temporary table

# PATIENT_NODES=pd.read_csv('data/msoa_postcodes_truncated.csv') # List of postcodes (not currently used)
ADMISSIONS = np.loadtxt('data/LSOA_strokes_no_index.csv',
                        delimiter=',')  # Admissions for each patient node.  PREVIOUSLY using "msoa_truncated_stemi.csv"
TOTAL_ADMISSIONS = np.sum(ADMISSIONS)
TRAVEL_MATRIX = np.loadtxt('data/stroke_time_matrix_data_only.csv',
                           delimiter=',')  # Node to hospital matrix.  PREVIOUSLY using "msoa_truncated_dist_matrix.csv"

## Initialise variables
GENERATIONS = 300
INITIAL_RANDOM_POPULATION_SIZE = 500
NEW_RANDOM_POPULATION_EACH_GENERATION = 0.05  # number of new random mmers each generation as proportion of selected population
MINIMUM_SELECTED_POPULATION_SIZE = 500
MAXIMUM_SELECTED_POPULATION_SIZE = 500 # Using different min and max allows for flexible population size within prescribed limits
MUTATION_RATE = 0.002  # prob of each gene mutationg each generation
TARGET_ADMISSIONS = 600
SKIP_BREEDING_IN_FIRST_GENERATION = 0  # Ehen set as 1, used to just select pareto front from initial population (without breeding first)
population = np.zeros((0, HOSPITAL_COUNT)) # initialise population array
CROSSOVERPROB = 100  # % likely to happen
USE_CROWDING = 0  # when set at 1 pareto front is reduced by crowding distance tournament, else use random thinning
CROSSOVERUNIFORM = False  # each gene has equal chance coming from each parent
if CROSSOVERUNIFORM == False:  # use standard cross over method, define number of cross over locations
    MAXCROSSOVERPOINTS = 3
CALCULATE_HAMMING = True # Monitor Hamming distance of breeding population. Memory hungry for large popualtions.

# Set up Epsilon if used as way of deteerminging no improvement in population
# Set to 0 for just max, 1 for just min, 2 for min and max, 3 for none
WHICH_EPSILON = 3
# the number of generations with the same epsilon to determine ending the algorithm as it's not improving
CONSTANT_EPSILON = 10
# stores the epsilon for the generation
epsilonmin = np.zeros(GENERATIONS)
epsilonmax = np.zeros(GENERATIONS)
# gets set to TRUE if the Epsilon (min, max, both) has been static for more than CONSTANT_EPSILON generations
static_epsilon = False

## Set up score normalisation array
# Set array to normalise all scores 0-1 (pareto needs higher=better).
# Absolute values not critical but high/low is. First number will score zero. Second number will score 1
# There is no truncation of scores
NORM_MATRIX = np.array(
    [[HOSPITAL_COUNT, 1], [250, 0], [250, 0], [TOTAL_ADMISSIONS, 1], [1, TOTAL_ADMISSIONS], [25, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 100]])

GA_DIVERSITY_PROGRESS = np.zeros((GENERATIONS,
                                  6))  # Store key information per generation to monitor how diverse the population is, and how strong it is
# i=Generation number
# (i,1) Size of first pareto front

# The following are recording a measure of 'diversity'
# (i,2) Hamming distance of the breeding population
# (i,3) Hamming distance of first pareto front

# The following are recording a measure of 'progress'
# (i,4) Max number for % patients attending unit within 30 mins, having 600 admissions [column 10 in score].. kp chose that to be from first Pareto
# (i,5) Epsilon
# (i,6) Hypervolume

## Generate initial random population and add to any loaded population
if INITIAL_RANDOM_POPULATION_SIZE > 0:
    population = ga.generate_random_population(INITIAL_RANDOM_POPULATION_SIZE, HOSPITAL_COUNT)

if LOAD_INITAL_POPULATION == 1:
    population = np.vstack((LOAD_POPULATION, population))
### Fix open all necessary hospitals
if DO_FIX_HOSPITAL_STATUS == 1:
    # Takes the 10th column from the hospital.csv file and if "1" then open, "-1" then closed
    population = ga.fix_hospital_status(population, HOSPITAL_STATUS)

population_size = len(population[:, 0])  # size of starting population

print('Begin generations: ', datetime.datetime.now().strftime("%H:%M:%S"), 'Starting population size: ',
      population_size)
# KP DO: SAVE FIRST GENERATION TO DETECT PROGRESS ON THE LASTEST SAVED PARETO FRONT
## Genetic algorithm. Breed then use pareto selection
for generation in range(GENERATIONS):
    if not static_epsilon:  # when Epsilon is static, then stop breeding - the idea is that you've found a good stabel populaiton that will not improve further
        # Being static is dependant on: number of generation choose to need to be the same (CONSTANT_EPSILON), and which epsilon to be static (max: take all the worst attributes for each person, use the best of these. min: take all the worst attributes for each person, use the worst of these. Or need both min and max to be the same)
        # hamming distance of population (before randoms added) and before breeding
        if CALCULATE_HAMMING:
            hamming_distances = pdist(population, 'hamming')  # the proportion which disagree
            average_hamming = np.average(hamming_distances)
        else:
            average_hamming = 999
        GA_DIVERSITY_PROGRESS[
            generation, 2] = average_hamming  # "AVERAGE HAMMING DISTANCE, 1=most diverse, 0=homogenous

        # Add in random population if required, as fraction of total population (adult+child)
        if NEW_RANDOM_POPULATION_EACH_GENERATION > 0:
            population_size = len(population[:, 0])
            new_random_population = ga.generate_random_population(
                int(NEW_RANDOM_POPULATION_EACH_GENERATION * population_size), HOSPITAL_COUNT)
            ### Fix open or closed all necessary hospitals
            if DO_FIX_HOSPITAL_STATUS == 1:
                new_random_population = ga.fix_hospital_status(new_random_population, HOSPITAL_STATUS)

            population = np.vstack((population, new_random_population))  # add random population to current population

        ### Remove all zero rows, and remove non-unique rows
        check_hospitals = np.sum(population, axis=1) > 0
        population = population[check_hospitals, :]
        population = ga.unique_rows(population)

        population_size = len(
            population[:, 0])  # current population size. Number of children generated will match this.

        if generation > 0 or SKIP_BREEDING_IN_FIRST_GENERATION != 1:  # used to skip breeding in first generation if wanted
            child_population = np.zeros((int(population_size / 2) * 2,
                                         HOSPITAL_COUNT))  # create empty matrix for children (ensure this is an even number)
            #        for mating in range (0,int(population_size/2)*2,2): # each mating will produce two children (the mating count jumps in steps of 2)
            for mating in range(0, int(population_size / 2) * 2,
                                2):  # each mating will produce two children (the mating count jumps in steps of 2)
                parent1_ID = rn.randint(0, population_size - 1)  # select parent ID at random
                parent2_ID = rn.randint(0, population_size - 1)  # select parent ID at random
                parent = np.vstack((population[parent1_ID, :], population[parent2_ID, :]))  # chromosome of parent 1 & 2
                crossoveroccur = rn.randint(0,
                                            100)  # print("Probability of crossover occuring is", CROSSOVERPROB ,".  This go it's", crossoveroccur,".  If < then CROSSOVER, else NO CROSSOVER AND PARENTS MAKE IT THROUGH TO NEXT GENERATION")
                if crossoveroccur < CROSSOVERPROB:
                    if CROSSOVERUNIFORM == True:  # USING 2 PARENTS,  CREATE CHILD1 AND CHILD2 USING CROSSOVER
                        # UNIFORM CROSSOVER: EACH GENE HAS EQUAL CHANCE TO COME FROM EACH PARENT
                        child = ga.f_uniform_crossover(parent, HOSPITAL_COUNT)
                    else:
                        # CROSSOVER HAPPENS IN A SET NUMBER OF LOCATIONS (THE MAXIMUM NUMBER OF LOCATION IS USER DEFINED).  MIN=1
                        child = ga.f_location_crossover(parent, MAXCROSSOVERPOINTS, HOSPITAL_COUNT)
                else:
                    # NO CROSSOVER, PARENTS GO TO NEXT GENERATION
                    child = parent
                child_population[mating:mating + 2, :] = child
            ### Random mutation
            random_mutation_array = np.random.random(
                size=(len(child_population[:, 0]), HOSPITAL_COUNT))  # random numbers 0-1
            random_mutation_array[random_mutation_array <= MUTATION_RATE] = 1  # set all values not to be mutated to 1
            random_mutation_array[random_mutation_array < 1] = 0  # set all other values to zero
            child_population = (1 - random_mutation_array) * child_population + (
            random_mutation_array * (1 - child_population))  # Inverts zero or one when mutation array value is one

            ### Fix open or closed all necessary hospitals
            if DO_FIX_HOSPITAL_STATUS == 1:
                child_population = ga.fix_hospital_status(child_population, HOSPITAL_STATUS)

            # Add child population to adult population    
            population = np.vstack((population, child_population))

            ### Remove all zero rows, and remove non-unique rows
        check_hospitals = np.sum(population, axis=1) > 0
        population = population[check_hospitals, :]
        population = ga.unique_rows(population)
        # np.savetxt(OUTPUT_LOCATION+'/whole_population_after_breeding_and_randoms.csv',population,delimiter=',',newline='\n')

        #  population_size=len(population[:,0])

        # Select pareto front
        unselected_population = population
        population = np.zeros((0, HOSPITAL_COUNT))
        population_size = 0
        (unselected_scores, hospital_admissions_matrix) = ga.score(unselected_population, TARGET_ADMISSIONS,
                                                                   TRAVEL_MATRIX, ADMISSIONS, TOTAL_ADMISSIONS,
                                                                   pareto_include, generation == GENERATIONS - 1,
                                                                   nscore_parameters)
        GA_DIVERSITY_PROGRESS[generation, 4] = np.amax(unselected_scores[:,
                                                       10])  # The max number for % patients attending unit within 30 mins, having 600 admissions [column 10 in score].. kp chose that to be from first Pareto
        scores = np.zeros((0, nscore_parameters))
        store_pareto = True

        # The following either 
        # 1) adds more members, from successively lower pareto fronts (from the remaining unselected population), if population size not large enough
        # Or 2) if population size exceeds the MAXIMUM_SELECTED_POPULATION_SIZE then solutions are pruned using crowding_selection (choosing members with the largest crowding distance) or thinning (at random)

        # Keep adding Pareto fronts until minimum population size is met

        #    #hamming distance of whole population before pareto (after breeding)
        #    hamming_distances=pdist(unselected_population,'hamming')#the proportion which disagree
        #    GA_DIVERSITY_PROGRESS[generation,2]=np.average(hamming_distances) #"AVERAGE HAMMING DISTANCE, 1=most diverse, 0=homogenous

        while population_size < MINIMUM_SELECTED_POPULATION_SIZE:
            max_new_population = MAXIMUM_SELECTED_POPULATION_SIZE - population_size  # new maximum population size to add
            norm_unselected_scores = ga.normalise_score(unselected_scores,
                                                        NORM_MATRIX)  # Set array to normalise all scores 0-1 (pareto needs higher=better)
            #        norm_scores=ga.normalise_score(unselected_scores,NORM_MATRIX) # normalise scores
            score_matrix_for_pareto = norm_unselected_scores[:,
                                      pareto_include]  # select scores to use in Pareto selection
            # np.savetxt(OUTPUT_LOCATION+'/pre_pareto_scores.csv',score_matrix_for_pareto,delimiter=',',newline='\n')
            pareto_index = ga.pareto(score_matrix_for_pareto)  # Pareto selection
            new_pareto_front_population = unselected_population[pareto_index, :]  # New Pareto population
            new_pareto_front_scores = unselected_scores[pareto_index, :]  # New Pareto population scores
            np.savetxt(OUTPUT_LOCATION + '/pareto_scores.csv', new_pareto_front_scores, delimiter=',', newline='\n')

            if store_pareto:
                # Save the first Pareto front (lower, or pruned, Pareto Fronts may be used in the next breeding population but are not stored here)
                # np.savetxt(OUTPUT_LOCATION+'/scores.csv',new_pareto_front_scores,delimiter=',',newline='\n')
                # np.savetxt(OUTPUT_LOCATION+'/hospitals.csv',new_pareto_front_population,delimiter=',',newline='\n')
                scores_hospitals = np.hstack((new_pareto_front_scores, new_pareto_front_population))
                np.savetxt(OUTPUT_LOCATION + '/scores_hospitals.csv', scores_hospitals, delimiter=',', newline='\n')
                new_pareto_front_hospital_admissions = hospital_admissions_matrix[pareto_index,
                                                       :]  # New Pareto population scores
                np.savetxt(OUTPUT_LOCATION + '/admissions.csv', new_pareto_front_hospital_admissions, delimiter=',',
                           newline='\n')
                # np.savetxt(OUTPUT_LOCATION+'/epsilon min.csv',epsilonmin,delimiter=',',newline='\n')
                # np.savetxt(OUTPUT_LOCATION+'/epsilon max.csv',epsilonmax,delimiter=',',newline='\n')
                store_pareto = False
                GA_DIVERSITY_PROGRESS[generation, 1] = len(new_pareto_front_population[:, 0])
                # hamming distance of first pareto front
                if CALCULATE_HAMMING:
                    hamming_distances = pdist(new_pareto_front_population, 'hamming')  # the proportion which disagree
                    average_hamming = np.average(hamming_distances)
                else:
                    average_hamming = 999
                # "AVERAGE HAMMING DISTANCE, 1=most diverse, 0=homogenous
                GA_DIVERSITY_PROGRESS[generation, 3] = average_hamming
                np.savetxt(OUTPUT_LOCATION + '/hamming.csv', GA_DIVERSITY_PROGRESS, delimiter=',', newline='\n')

            # Selecting another Pareto front may take population above maximum size, in which latest add population is reduced
            if len(new_pareto_front_population[:, 0]) > max_new_population:
                # Pick subset.  Either by crowding distance (based on level of clustering of points), or thinning
                if USE_CROWDING == 1:  # Crowding distance
                    (new_pareto_front_population, new_pareto_front_scores) = ga.crowding_selection(
                        new_pareto_front_population, new_pareto_front_scores, max_new_population)
                else:  # use thinning (selecting population to be removed at random so within the right population limit)
                    pick_list = np.zeros(len(new_pareto_front_population[:, 0]))
                    pick_list[
                    0:max_new_population] = 1  # Add required number of 1s (to indicate that these are to be removed from the population)
                    np.random.shuffle(pick_list)
                    new_pareto_front_population = new_pareto_front_population[pick_list == 1, :]
                    new_pareto_front_scores = new_pareto_front_scores[pick_list == 1, :]

            else:  # Identify remaining unselected population
                unselected_index = np.logical_not(pareto_index)
                unselected_population = unselected_population[unselected_index, :]
                unselected_scores = unselected_scores[unselected_index, :]
            # Add latest selection (population and scores) to previous selection, and remeaure size 
            population = np.vstack((population, new_pareto_front_population))
            scores = np.vstack((scores, new_pareto_front_scores))
            population_size = len(population[:, 0])

        max_result = np.amax(scores[:, 10])  # Use to display progress
        # Calculate Epsilon (measure of how good the population is attaining the reference point (this is all with value 1 after normalising the scores))
        norm_scores = ga.normalise_score(scores, NORM_MATRIX)

        if WHICH_EPSILON < 3:  # =3 means don't do an epsilon check
            # When epsilon is static for CONSTANT_EPSILON generations, then static_epsilon is set to True and so will jump over the bulk of the code for the remaining generations
            score_matrix_for_epsilon = norm_scores[:, pareto_include]  # select scores to use in Pareto selection
            np.savetxt('output/pre_epsilon_scores.csv', score_matrix_for_epsilon, delimiter=',', newline='\n')
            (epsilonmin[generation], epsilonmax[generation]) = ga.calculate_epsilon(
                score_matrix_for_epsilon)  # (nscore_parameters,score_matrix_for_epsilon,population_size)
            if generation >= CONSTANT_EPSILON - 1:  # only check if there's enough prior generation to go through
                static_epsilon = ga.is_epsilon_static(epsilonmin, epsilonmax, generation, CONSTANT_EPSILON,
                                                      WHICH_EPSILON)

        # At end of generation print time, generation and population size (for monitoring)
        print(datetime.datetime.now().strftime("%H:%M:%S"), 'Generation: ', generation + 1, ' Population size: ',
              population_size, ' Hamming: ', average_hamming, ' Max result: ', max_result)
