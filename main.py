import numpy as np
import client
from tabulate import tabulate
import math
import random

key = 'zM16c5GKsq8Owcb4e3wYDxdX8i97OqcERSvC5jgWJTSFQ94Lot'
pop_size = 10
select_sure = 3
cross_select_from = 8
overfit_weights =[ 0.00000000e+00, -1.31246059e-12, -2.56689599e-13 , 3.47367648e-11,
 -1.42907754e-10, -7.08379045e-16,  6.12137257e-16,  2.46783380e-05,
 -1.63862328e-06, -1.50719802e-08 , 7.52797544e-10]

def formatArray(x):
    y=[]
    for i in range (0, pop_size):
        y.append([x[i]])

    return np.array(y)

def mutateall(temp,prob, mutate_range):
    vector = np.copy(temp)
    it = 0
    while it < len(vector):
        fact = random.uniform(-mutate_range, mutate_range)
        it += 1
        tp = vector[it-1]*(fact+1)
        vector[it-1] = np.random.choice([tp, vector[it-1]], p=[prob,1-prob])
        if abs(vector[it-1])>10:
            if(vector[it-1]<0):
                vector[it-1] = -10
            else:
                vector[it-1] = 10
    
    return vector
iter = 15
def crossover(vector1, vector2, mutate_range,prob_mut_cross, index=-1):
    send1, send2 = (vector1.tolist(), vector2.tolist())

    a = np.random.choice(np.arange(0, 11),5, replace=False)
    lis = [i for i in a.tolist()]
    it = 0
    for i in lis:
        send1[i], send2[i] = (np.copy(vector2[i]), np.copy(vector1[i]))
        it += 1

    return mutateall(send1,prob_mut_cross,mutate_range), mutateall(send2,prob_mut_cross,mutate_range),send1,send2,a


def d2(var):
    return int(var/2)

def main():

    mutate_range=0.1
    prob_mut_cross = 0.9
    print("pop_size: ", pop_size)
    print("iter: ", iter)
    print("cross_select_from ",cross_select_from)
    print("select_sure",select_sure)
    print("prob_mut_cross",prob_mut_cross)
    print("mutate_range",mutate_range)

    to_send = [-20 for i in range(11)]

    min_error, min_error1, min_error2 = (-1, -1, -1)

    parenterrors, parenterrors1, parenterrors2 = (np.zeros(pop_size), np.zeros(pop_size), np.zeros(pop_size))
    population = np.zeros((pop_size, 11))

    # generate the population
    it2 = 0
    while it2 < pop_size:
        temp = np.copy(overfit_weights)
        it2 += 1
        population[it2-1] = np.copy(mutateall(temp,0.85,mutate_range))

    # generate errors for each individual in the population
    it2 = 0
    while it2 < pop_size:
        # passing a list to the get_errors function
        temp = population[it2].tolist()
        it2 += 1
        err = client.get_errors(key, temp)
        # adding the two errors and storing in parenterror - fitness function
        tp = err[0] + err[1]
        parenterrors[it2-1] = np.copy((err[0]+err[1]))
        # parenterrors[j] = np.copy((err[0]+err[1]))
        parenterrors1[it2-1], parenterrors2[it2-1] = (np.copy((err[0])), np.copy((err[1])))

    # have to change this to a while loop with appropriate condition later
    for iter_num in range(0, iter):

        if((iter_num)%6==0 and iter_num>0):
            mutate_range = mutate_range - 0.01
            prob_mut_cross = prob_mut_cross + 0.01
            print("::::::::::::::::changing ", mutate_range)



        #has popsize/2 pairs, each is a set of parents used to generate two children
        arrchoices=np.zeros((d2(pop_size),2))
        arrposswap=np.zeros((d2(pop_size),5))

        #has the array of all the children
        arrchildren=np.zeros((pop_size,11))
        arrchildrenmutated=np.zeros((pop_size,11))
        arrchilderrors=np.zeros((pop_size))

        print("\n\n\n\n******************ITERATION-"+str(iter_num)+"******************")

        parenerrorsinds = parenterrors.argsort()
        parenterrors, parenterrors1, parenterrors2  = (np.copy(parenterrors[parenerrorsinds[::1]]), np.copy(parenterrors1[parenerrorsinds[::1]]), np.copy(parenterrors2[parenerrorsinds[::1]]))
        population = np.copy(population[parenerrorsinds[::1]])

            #parents with their errors
        arrparents, arrparrerrs = (np.copy(population), np.copy(parenterrors))
        new_iter = 0
        # debug statements
        iterj = 0
        while(iterj < pop_size):
            iterj += 1
            print("human" + str(iterj-1)+":")
            print()
            print("error: " + str(parenterrors[iterj-1]))
            print()
            print("trainerror: " + str(parenterrors1[iterj-1]))
            print()
            print("validationerror: " + str(parenterrors2[iterj-1]))
            print()
            print("\tvalues: "+str(population[iterj-1]))
            print()
            print("_________________________________________________________________")
            print()

        child_population = np.zeros((pop_size, 11))
        it2 = 0
        while(new_iter < pop_size):

            # arr = crossover_select(parentprobalities)
            # TODO: Select randomly among top k parents  (For now k =10)
            arr = random.sample(range(8), 2)

            # Sending parents for crossover
            temp = crossover(population[arr[0]], population[arr[1]],mutate_range,prob_mut_cross)
            it2 += 1
            if temp[1].tolist() == population[arr[0]].tolist(): 
                continue
            elif temp[0].tolist() == population[arr[0]].tolist(): 
                continue
            elif temp[0].tolist() == population[arr[1]].tolist():
                continue
            elif temp[1].tolist() == population[arr[1]].tolist():
                continue
            
            for it in range(0, 2):
                arrchoices[d2(new_iter)][it]=np.copy(arr[it])
            
            arrposswap[d2(new_iter)]=np.copy(np.sort(temp[4]))

            arrchildren[new_iter], arrchildrenmutated[new_iter], child_population[new_iter]  = (np.copy(temp[2]), np.copy(temp[0]), np.copy(temp[0]))

            new_iter = new_iter + 1

            arrchildren[new_iter], arrchildrenmutated[new_iter], child_population[new_iter] = (np.copy(temp[3]), np.copy(temp[1]), np.copy(temp[1]))
            new_iter = new_iter + 1


        childerrors, childerrors1, childerrors2  = (np.zeros(pop_size), np.zeros(pop_size), np.zeros(pop_size))

        # generate errors for each child
        it2 = 0
        while it2 < pop_size:
            temp = child_population[it2].tolist()
            err = client.get_errors(key, temp)
            childerrors[it2], childerrors1[it2], childerrors2[it2] = (np.copy((err[0]+err[1])), np.copy((err[0])), np.copy((err[1])))
            it2 += 1
            arrchilderrors[it2-1]=np.copy(childerrors[it2-1])

        # Sort children
        childinds = np.copy(childerrors.argsort())
        childerrors, childerrors1, childerrors2 = (np.copy(childerrors[childinds[::1]]), np.copy(childerrors1[childinds[::1]]), np.copy(childerrors2[childinds[::1]]))
        child_population = np.copy(child_population[childinds[::1]])
        # TODO: Select the best select_sure number of parents and chilren [select these many parents and children for sure]

        # now the children are sorted and stored in child and parents are sorted in population
        # we will now create a tempbank array to store top k parents, top k childs and rest being sorted taking from the top
        tempbankerr, tempbankerr1, tempbankerr2, tempbank = (np.zeros(pop_size), np.zeros(pop_size), np.zeros(pop_size), np.zeros((pop_size, 11)))
        
        for j in range(0, select_sure):
            
            #choosing the top jth parent and putting in the array
            tempbank[j], tempbankerr[j], tempbankerr1[j], tempbankerr2[j] = (np.copy(population[j]), np.copy(parenterrors[j]), np.copy(parenterrors1[j]), np.copy(parenterrors2[j]))
            
            #choosing the top jth child and putting it into the array 
            tempbank[j+select_sure], tempbankerr[j+select_sure], tempbankerr1[j+select_sure], tempbankerr2[j+select_sure] = (np.copy(child_population[j]),np.copy(childerrors[j]), np.copy(childerrors1[j]), np.copy(childerrors2[j]))
            

        # combining parents and children into one array
        # TODO: Concatenating remaining parents and children and selecting from them
        candidates, candidate_errors = (np.copy(np.concatenate([population[select_sure:], child_population[select_sure:]])), np.copy(np.concatenate([parenterrors[select_sure:], childerrors[select_sure:]])))
        candidate_errors1 , candidate_errors2 = (np.copy(np.concatenate([parenterrors1[select_sure:], childerrors1[select_sure:]])), np.copy(np.concatenate([parenterrors2[select_sure:], childerrors2[select_sure:]])))

        # sorting all the candidates by error
        candidate_errors_inds = candidate_errors.argsort()
        candidate_errors, candidate_errors1, candidate_errors2 = (np.copy(candidate_errors[candidate_errors_inds[::1]]), np.copy(candidate_errors1[candidate_errors_inds[::1]]), np.copy(candidate_errors2[candidate_errors_inds[::1]]))
        candidates = np.copy(candidates[candidate_errors_inds[::1]])
        # TODO: Select the best popsize - 2*(select_sure)
        cand_iter = 0
        selec2 = 2*select_sure 

        while(cand_iter + selec2 < pop_size):
            tempbank[cand_iter+selec2], tempbankerr[cand_iter+selec2]  = (np.copy(candidates[cand_iter]), np.copy(candidate_errors[cand_iter]))
            tempbankerr1[cand_iter+selec2], tempbankerr2[cand_iter+selec2] = (np.copy(candidate_errors1[cand_iter]),  np.copy(candidate_errors2[cand_iter]))
            cand_iter = cand_iter + 1


        #now setting the next population
        parenterrors, parenterrors1, parenterrors2 = (np.copy(tempbankerr), np.copy(tempbankerr1), np.copy(tempbankerr2))
        population=np.copy(tempbank)
        min_parent_error = min_error
        #we will now sort before updating min_error
        parenerrorsinds = parenterrors.argsort()
        check_error = np.copy(parenterrors[parenerrorsinds[::1]])
        parent_error = np.zeros((pop_size,3))
        population = np.copy(population[parenerrorsinds[::1]])
        parenterrors, parenterrors1, parenterrors2  = (np.copy(parenterrors[parenerrorsinds[::1]]), np.copy(parenterrors1[parenerrorsinds[::1]]), np.copy(parenterrors2[parenerrorsinds[::1]]))


       

        # showtable(arrparents,arrparrerrs,arrchoices,arrchildren,arrchildrenmutated,arrchilderrors,arrposswap)
        arrparents, arrchildren, arrchildrenmutated = (formatArray(arrparents), formatArray(arrchildren), formatArray(arrchildrenmutated))

        tempchoice = []
        for i in range(0, pop_size):
            tempchoice.append([["P" + str(int(arrchoices[d2(1)][0])),"P" + str(int(arrchoices[d2(i)][1]))]])
        tempchoice=np.array(tempchoice)
        
        tempswap = []
        for i in range(d2(pop_size)):
            for j in range(0, 2):
                tempswap.append([arrposswap[i]])
        tempswap=np.array(tempswap)


        final1, final2, final3 = (np.zeros((pop_size,3),dtype=object), np.zeros((pop_size,4),dtype=object), np.zeros((pop_size,3),dtype=object))

        for i in range(0, pop_size):
            final1[i][0], final1[i][1], final1[i][2] = ("P"+str(i), arrparents[i], arrparrerrs[i])
        print()
        print()
        for i in range(0, pop_size):
            final2[i][0], final2[i][1], final2[i][2], final2[i][3] = ("C"+str(i), tempchoice[i], tempswap[i], arrchildren[i])
        print()
        print()
        for i in range(0, pop_size):
            final3[i][0], final3[i][1], final3[i][2] = ("M"+str(i), arrchildrenmutated[i], arrchilderrors[i])

        if(min_error == -1 or parenterrors[0] < min_error ):
            nochange=0
            to_send = np.copy(population[0])
            min_error, min_error1, min_error2 = (np.copy(parenterrors[0]), np.copy(parenterrors1[0]), np.copy(parenterrors2[0]))

        else:
            print("No improvement")
            nochange = nochange + 1
        print()
        print()
        print("Min error = ", min_error)
        print()
        print()
        print("Train error = ", min_error1)
        print()
        print()
        print("Validation error = ", min_error2)
        print()
        print()

        headers1, headers2, headers3 = (["Index","Population", "Population Errors"], ["Index","Parents", "Crossover positions","Children"], ["Index","Mutated Children","Mutated Children Errors"])
        table1, table2, table3 = (tabulate(final1, headers1, tablefmt="fancy_grid"), tabulate(final2, headers2, tablefmt="fancy_grid"), tabulate(final3, headers3, tablefmt="fancy_grid"))
        # printing tables
        #print(table1)
        #print(table2)
        #print(table3)
    return to_send


to_send = main()
print("__________________________________________________________________________________")
print("sending the vector:\n\n"+str(to_send)+"\n\nwas it successfully submitted?\n",
    client.submit(key, to_send.tolist()))
print("Code has ended")
print("__________________________________________________________________________________")
print()
print()
