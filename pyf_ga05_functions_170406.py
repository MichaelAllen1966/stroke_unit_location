import numpy as np
import random as rn

def calculate_crowding(scores):
    # Crowding is based on chrmosome scores (not chromosome binary values)
    # All scores are normalised between low and high
    # For any one score, all solutions are sorted in order low to high
    # Crowding for chromsome x for that score is the difference between th enext highest and next lowest score
    # Total crowding value sums all crowding for all scores
    population_size=len(scores[:,0])
    number_of_scores=len(scores[0,:])
    # create crowding matrix of population (row) and score (column)
    crowding_matrix=np.zeros((population_size,number_of_scores)) 
    # normalise scores
    normed_scores = (scores-scores.min(0))/scores.ptp(0) # numpy ptp is range (max-min)
    # Calculate crowding
    for col in range(number_of_scores): # calculate crowding distance for each score in turn
        crowding=np.zeros(population_size) # One dimensional array
        crowding[0]=1 # end points have maximum crowding
        crowding[population_size-1]=1 # end points have maximum crowding
        sorted_scores=np.sort(normed_scores[:,col]) # sort scores
        sorted_scores_index=np.argsort(normed_scores[:,col]) # index of sorted scores
        crowding[1:population_size-1]=sorted_scores[2:population_size]-sorted_scores[0:population_size-2] # crowding distance
        re_sort_order=np.argsort(sorted_scores_index) # re-sort to original order step 1
        sorted_crowding=crowding[re_sort_order] # re-sort to orginal order step 2
        crowding_matrix[:,col]=sorted_crowding # record crowding distances
    crowding_distances=np.sum(crowding_matrix,axis=1) # Sum croding distances of all scores
    return crowding_distances
    
def crowding_selection(population,scores,number_to_select):
    # This function selects a number of solutions based on tournament of crowding distances
    # Two members of the population ar epicked at random
    # The one with the higher croding dostance is always picked
    crowding_distances=calculate_crowding(scores) # crowding distances for each member of the population
    picked_population=np.zeros((number_to_select,len(population[0,:]))) # array of picked solutions (actual solution not ID)
    picked_scores=np.zeros((number_to_select,len(scores[0,:]))) # array of scores for picked solutions
    for i in range(number_to_select):
        population_size=len(population[:,0])
        fighter1ID=rn.randint(0,population_size-1) # 1st random ID
        fighter2ID=rn.randint(0,population_size-1) # 2nd random ID
        if crowding_distances[fighter1ID]>=crowding_distances[fighter2ID]: # 1st solution picked
            picked_population[i,:]=population[fighter1ID,:] # add solution to picked solutions array
            picked_scores[i,:]=scores[fighter1ID,:] # add score to picked solutions array
            # remove selected solution from available solutions
            population=np.delete(population,(fighter1ID), axis=0) # remove picked solution - cannot be chosen again
            scores=np.delete(scores,(fighter1ID), axis=0) # remove picked score (as line above)
            crowding_distances=np.delete(crowding_distances,(fighter1ID), axis=0) # remove crowdong score (as line above)
        else: # solution 2 is better. Code as above for 1st solution winning
            picked_population[i,:]=population[fighter2ID,:]
            picked_scores[i,:]=scores[fighter2ID,:]
            population=np.delete(population,(fighter2ID), axis=0)
            scores=np.delete(scores,(fighter2ID), axis=0)
            crowding_distances=np.delete(crowding_distances,(fighter2ID), axis=0)
    return (picked_population,picked_scores)

def generate_random_population(rows,cols):
    population=np.zeros((rows,cols)) # create array of zeros
    for i in range(rows):
        x=rn.randint(1,cols) # Number of 1s to add
        population[i,0:x]=1 # Add requires 1s
        np.random.shuffle(population[i]) # Shuffle the 1s randomly
    return population
   
def pareto(scores):
    # In this method the array 'scores' is passed to the function.
    # Scores have been normalised so that higher values dominate lower values.
    # The function returns a Boolean array identifying which rows of the array 'scores' are non-dominated (the Pareto front)
    # Method based on assuming everything starts on Pareto front and then records dominated points
    pop_size=len(scores[:,0])
    pareto_front=np.ones(pop_size,dtype=bool)
    for i in range(pop_size):
        for j in range(pop_size):
            if all (scores[j]>=scores[i]) and any (scores[j]>scores[i]):
                # j dominates i
                pareto_front[i]=0
                break
    return pareto_front
    
def normalise_score(score_matrix,norm_matrix):
    # normalise 'score matrix' with reference to 'norm matrix' which gives scores that produce zero or one
    norm_score=np.zeros(np.shape(score_matrix)) # create normlaises score matrix with same dimensions as original scores
    number_of_scores=len(score_matrix[0,:]) # number of different scores
    for col in range(number_of_scores): # normaise for each score in turn
        score_zero=norm_matrix[col,0]
        score_one=norm_matrix[col,1]
        score_range=score_one-score_zero
        norm_score[:,col]=(score_matrix[:,col]-score_zero)/score_range
    return norm_score

def score(population,TARGET_ADMISSIONS,TRAVEL_MATRIX,NODE_ADMISSIONS,TOTAL_ADMISSIONS,pareto_include,CALC_ALL,nscore_parameters):
#Only calculate the score that is needed by the pareto front, as determined by the array: pareto_include
#Unless CALC_ALL=True (set for the last generation) as then print out all the parameter values

    CALC_ALL=True # MA reporting all

    # Score_matrix:
    # 0: Number of hospitals
    # 1: Average distance
    # 2: Maximum distance
    # 3: Maximum admissions to any one hopsital
    # 4: Minimum admissions to any one hopsital
    # 5: Max/Min Admissions ratio
    # 6: Proportion patients within target distance 1
    # 7: Proportion patients within target distance 2
    # 8: Proportion patients within target distance 3
    # 9: Proportion patients attending unit with target admission numbers
    # 10: Proportion of patients meeting distance 1 (~30 min) and admissions target
    # 11: Proportion of patients meeting distance 2 (~45 min) and admissions target
    # 12: Proportion of patients meeting distance 3 (~60 min) and admissions target
    # 13: Clinical benefit, additional benefit per 100 treatable patients
    
    TARGET_DISTANCE_1=30 # straight line km, equivalent to 30 min
    TARGET_DISTANCE_2=45 # straight line km, equivalent to 45 min
    TARGET_DISTANCE_3=60 # straight line km, equivalent to 60 min

    pop_size=len(population[:,0]) # Count number of solutions to evaluate
    score_matrix=np.zeros((pop_size,nscore_parameters)) # Create an empty score matrix
    hospital_admissions_matrix=np.zeros((pop_size,len(TRAVEL_MATRIX[0,:])))#store teh hospital admissions, col = hospital, row = population
    for i in range(pop_size): # Loop through population of solutions
        node_results=np.zeros((len(NODE_ADMISSIONS),10))
        # Node results stores results by patient node. These are used in the calculation of results 
        # Node result smay be of use to export at later date (e.g. for detailed analysis of one scenario)
        # Col 0: Distance to closest hospital
        # Col 1: Patients within target distance 1 (boolean)
        # Col 2: Patients within target distance 2 (boolean)
        # Col 3: Patients within target distance 3 (boolean)
        # Col 4: Hospital ID
        # Col 5: Number of admissiosn to hospital ID 
        # Col 6: Does hospital meet admissions target (boolean)
        # Col 7: Admissions and target distance 1 both met (boolean)
        # Col 8: Admissions and target distance 2 both met (boolean)
        # Col 9: Admissions and target distance 3 both met (boolean)
        
        # Count hospitals in each solution
        if 0 in pareto_include or CALC_ALL:
            score_matrix[i,0]=np.sum(population[i])

        # Calculate average distance
        mask=np.array(population[i],dtype=bool)
        # hospital_list=np.where(mask) # list of hospitals in selection. Not currently used
        masked_distances=TRAVEL_MATRIX[:,mask]
        # Calculate results for each patient node
        node_results[:,0]=np.amin(masked_distances,axis=1) # distance to closest hospital
        node_results[:,1]=node_results[:,0]<=TARGET_DISTANCE_1 # =1 if target distance 1 met
        node_results[:,2]=node_results[:,0]<=TARGET_DISTANCE_2 # =1 if target distance 2 met
        node_results[:,3]=node_results[:,0]<=TARGET_DISTANCE_3 # =1 if target distance 3 met
        closest_hospital_ID=np.argmin(masked_distances,axis=1) # index of closest hospital. 
        node_results[:,4]=closest_hospital_ID # stores hospital ID in case table needs to be exported later, but bincount below doesn't work when stored in NumPy array (which defaults to floating decimal)
        # Create matrix of number of admissions to each hospital
        hospital_admissions=np.bincount(closest_hospital_ID,weights=NODE_ADMISSIONS) # np.bincount with weights sums
        hospital_admissions_matrix[i,mask]=hospital_admissions#putting the hospital admissions into a matrix with column per hospital, row per solution.  Used to output to sheet
        # record closest hospital (unused)
        node_results[:,5]=np.take(hospital_admissions,closest_hospital_ID) # Lookup admissions to hospital used 
        node_results[:,6]=node_results[:,5]>TARGET_ADMISSIONS # =1 if admissions target met
        
        # Calculate average distance by multiplying node distance * admission numbers and divide by total admissions
        if 1 in pareto_include or CALC_ALL:
            weighted_distances=np.multiply(node_results[:,0],NODE_ADMISSIONS)
            average_distance=np.sum(weighted_distances)/TOTAL_ADMISSIONS
            score_matrix[i,1]=average_distance
        
        # Max distance for any one patient
        if 2 in pareto_include or CALC_ALL:
            score_matrix[i,2]=np.max(node_results[:,0])
        
        # Max, min and max/min number of admissions to each hospital
        if 3 in pareto_include or CALC_ALL:
            score_matrix[i,3]=np.max(hospital_admissions)
        if 4 in pareto_include or CALC_ALL:
            score_matrix[i,4]=np.min(hospital_admissions)
        if 5 in pareto_include or CALC_ALL:
            score_matrix[i,5]=score_matrix[i,3]/score_matrix[i,4]

        # Calculate proportion patients within target distance/time
        if 6 in pareto_include or CALC_ALL:
            score_matrix[i,6]=np.sum(NODE_ADMISSIONS[node_results[:,0]<=TARGET_DISTANCE_1])/TOTAL_ADMISSIONS
        if 7 in pareto_include or CALC_ALL:
            score_matrix[i,7]=np.sum(NODE_ADMISSIONS[node_results[:,0]<=TARGET_DISTANCE_2])/TOTAL_ADMISSIONS
        if 8 in pareto_include or CALC_ALL:
            score_matrix[i,8]=np.sum(NODE_ADMISSIONS[node_results[:,0]<=TARGET_DISTANCE_3])/TOTAL_ADMISSIONS

        # Calculate proportion patients attending hospital with target admissions
        if 9 in pareto_include or CALC_ALL:
            score_matrix[i,9]=np.sum(hospital_admissions[hospital_admissions>=TARGET_ADMISSIONS])/TOTAL_ADMISSIONS
        if 10 in pareto_include or CALC_ALL:
            # Sum patients who meet distance taregts
            node_results[:,7]=(node_results[:,1]+node_results[:,6])==2 # true if admissions and target distance 1 both met
            sum_patients_addmissions_distance1_met=np.sum(NODE_ADMISSIONS[node_results[:,7]==1])
            score_matrix[i,10]=sum_patients_addmissions_distance1_met/TOTAL_ADMISSIONS
        if 11 in pareto_include or CALC_ALL:
            # Sum patients who meet distance taregts
            node_results[:,8]=(node_results[:,2]+node_results[:,6])==2 # true if admissions and target distance 2 both met
            sum_patients_addmissions_distance2_met=np.sum(NODE_ADMISSIONS[node_results[:,8]==1])
            score_matrix[i,11]=sum_patients_addmissions_distance2_met/TOTAL_ADMISSIONS
        if 12 in pareto_include or CALC_ALL:
            # Sum patients who meet distance taregts
            node_results[:,9]=(node_results[:,3]+node_results[:,6])==2 # true if admissions and target distance 3 both met
            sum_patients_addmissions_distance3_met=np.sum(NODE_ADMISSIONS[node_results[:,9]==1])
            score_matrix[i,12]=sum_patients_addmissions_distance3_met/TOTAL_ADMISSIONS
        
        #Calculate clinical benefit: Emberson and Lee
        #Use 115 mins for the onset til travelling in ambulance (30 mins onset to call + 40 mins call to travel + 45 mins door to needle) + ? travel time (as deterined by the combination of hospital open)
#        if 13 in pareto_include or CALC_ALL:
#            onset_to_treatment_time = distancekm_to_timemin(node_results[:,0])+115
#            #constant to be used in the equation
#            factor=(0.2948/(1 - 0.2948))
#            #Calculate the adjusted odds ratio
#            clinical_benefit=np.array(factor*np.power(10, (0.326956 + (-0.00086211 * onset_to_treatment_time))))
#            # Patients that exceed the licensed onset to treatment time, set to a zero clinical benefit
#            clinical_benefit[onset_to_treatment_time>270]=0 
#            #Probabilty of good outcome per node
#            clinical_benefit = (clinical_benefit / (1 + clinical_benefit)) - 0.2948
#            #Number of patients with a good outcome per node
#            clinical_benefit = clinical_benefit*NODE_ADMISSIONS
#            score_matrix[i,13]=np.sum(clinical_benefit)/TOTAL_ADMISSIONS *100

        #hospital_admissions_matrix[i,:]=np.transpose(hospital_admissions)#putting the column into a row in the matrix
        #np.savetxt('output/admissions_test.csv',hospital_admissions_matrix[i,:],delimiter=',',newline='\n')
        
    return (score_matrix,hospital_admissions_matrix)
    
def unique_rows(a): # stolen off the interwebs
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    
def fix_hospital_status(l_population,l_HOSPITAL_STATUS):
    #Takes the 5th column from the hospital.csv file and if "1" then open, "-1" then closed
    HOSPITAL_STATUS_POPULATION=np.repeat(l_HOSPITAL_STATUS,len(l_population[:,0]),axis=0)#repeat the row "len(child_population[:,0])" number of times, so have 1 per solution row (matching the size of the child_population matrix)
    l_population[HOSPITAL_STATUS_POPULATION==1]=1 # Fixes the open hospitals to have a value 1
    l_population[HOSPITAL_STATUS_POPULATION==-1]=0 # Fixes the closed hospitals to have a value 0
    return l_population
    
    
def f_location_crossover(l_parent, l_MAXCROSSOVERPOINTS,l_CHROMOSOMELENGTH):
   
    number_crossover_points=rn.randint(1,l_MAXCROSSOVERPOINTS) # random, up to max
    crossover_points=rn.sample(range(1,l_CHROMOSOMELENGTH), number_crossover_points) # pick random crossover points in gene, avoid first position (zero position)
    crossover_points=np.append([0],np.sort(crossover_points)) # zero appended at front for calucation of interval to first crossover
    intervals=crossover_points[1:]-crossover_points[:-1] # this gives array of number of ones, zeros etc in each section.
    intervals=np.append([intervals],[l_CHROMOSOMELENGTH-np.amax(crossover_points)]) # adds in last interval of last cross-over to end of gene
    
    # Build boolean arrays for cross-overs
    current_bool=True # sub sections will be made up of repeats of boolean true or false, start with true

    # empty list required for append
    selection1=[] 

    for interval in intervals: # interval is the interval between crossoevrs (stored in 'intervals')
        new_section=np.repeat(current_bool,interval) # create subsection of true or false
        current_bool=not current_bool # swap true to false and vice versa
        selection1=np.append([selection1],[new_section]) # add the new section to the existing array
    
    selection1=np.array([selection1],dtype=bool) # **** not sure why this is needed but selection1 seems to have lost boolean nature
    selection2=np.invert(selection1) #  invert boolean selection for second cross-over product

    crossover1=np.choose(selection1,l_parent) # choose from parents based on selection vector
    crossover2=np.choose(selection2,l_parent)

    children=np.append(crossover1,crossover2,axis=0)
    
    return(children)
  
def f_uniform_crossover(l_parent, l_CHROMOSOMELENGTH):
#UNIFORM CROSSOVER MEANS EACH GENE HAS EQUAL CHANCE TO COME FROM EACH PARENT
    fromparent1=np.random.random_integers(0,1,(1,l_CHROMOSOMELENGTH)) # create array of 1 rows and chromosome columns and fill with 0 or 1 for which parent to take the gene from
    fromparent1=np.array(fromparent1,dtype=bool)
    fromparent2=np.invert(fromparent1)#opposite of fromparent1
    crossover1=np.choose(fromparent1,l_parent) # choose from the 2 parents based on select vector
    crossover2=np.choose(fromparent2,l_parent)
    children=np.append(crossover1,crossover2,axis=0)
    return(children)

def distancekm_to_timemin(distancekm):
#    Using EV's 5th order polynomial
#    speed=np.array(25.9364 + 0.740692*distance -0.00537274*np.power(distance,2)+ 0.000019121*np.power(distance,3)-0.0000000319161*np.power(distance,4)+0.0000000000199508*np.power(distance,5))
#    Using EV's 8th order polynomial, max distance is 400 (set all speeds for distances over 400, to if 400), otherwise the equation behaves in an odd way
    distancekm[distancekm>400]=400
    speedkmhr=np.array(17.5851+1.41592*distancekm-0.0212389*(distancekm**2)+0.000186836*(distancekm**3)-0.000000974605*(distancekm**4)+0.0000000030356*(distancekm**5)-0.00000000000551583*(distancekm**6)+0.00000000000000537742*(distancekm**7)-0.00000000000000000216925*(distancekm**8))
    timemin=np.array((distancekm/speedkmhr)*60)

#    time=np.array(25.9364 + 0.740692*distance -0.00537274*math.pow(distance,2)+ 1.9121e-05*math.pow(distance,3) -3.19161e-08*math.pow(distance,4)+1.99508e-11*math.pow(distance,5))
#    Using EV's 1st order polynomial
#    time=np.array(52.0884 + 0.0650975*distance)
    return(timemin)   

def calculate_epsilon(l_normscores):
#did use three inputs: (l_nscores,l_normscores,l_population):

#l_WHICH_EPSILON: Set to 0 for just max, 1 for just min, 2 for min and max, 3 for none

# The reference point for the Epsilon calculation "Utopia" in this instance is 1 for all dimensions.
#    utopia=np.ones(l_nscores*POPULATION).reshape((POPULATION,l_nscores))

#The algotihm requires all the scores to be divided by the utopia, but while it's 1 then no need.  Rows of code included below for completelness for when utopia <>1, but at the mo not required.
#    npdivide = l_normscores/utopia
#    epsilon = np.amin(npdivide,1)#the min score for each of the population.  Get each person to put forward their worst attribute
    epsilon = np.amin(l_normscores,1)#the min score for each of the population.  Get each person to put forward their worst attribute
    return (np.amin(epsilon),np.amax(epsilon))

def is_epsilon_static(l_epsilonmin, l_epsilonmax,l_generation,l_CONSTANT_EPSILON,l_WHICH_EPSILON):
    if l_WHICH_EPSILON==0:
        #need just epsilon max not change over CONSTANT_EPSILON generations
        return check_epsilon_static(l_epsilonmax,l_generation,l_CONSTANT_EPSILON)
    elif l_WHICH_EPSILON==1:
        #need just epsilon min not change over CONSTANT_EPSILON generations
        return check_epsilon_static(l_epsilonmin,l_generation,l_CONSTANT_EPSILON)
    elif l_WHICH_EPSILON==2:
        #need both epsilon min and epsilon max to not change
        if check_epsilon_static(l_epsilonmax,l_generation,l_CONSTANT_EPSILON):
            #only need to check epsilonmin if epsilonmax is satisfied
            return check_epsilon_static(l_epsilonmin,l_generation,l_CONSTANT_EPSILON)
        else:
            return False            

def check_epsilon_static(l_epsilonarray,l_generation,l_CONSTANT_EPSILON):
    i=0
    while l_epsilonarray[l_generation-i]==l_epsilonarray[l_generation-i-1] and i<l_CONSTANT_EPSILON:
       i+=1 
   
    if i==l_CONSTANT_EPSILON:
        #reached static population for 3 generations.  KP check: Can I set generation to GENERATIONS?
        #else do a while on a boolean
        return True #jumps over code for remaining generation
    else:
        return False #jumps over code for remaining generation