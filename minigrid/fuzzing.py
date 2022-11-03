from copy import copy, deepcopy
import random
import numpy as np
import time
from minigrid.minigrid_env import MiniGridEnv
random.seed(1)

def get_fitness(game, trace):
    game.start(render=False)
    #time.sleep(3)
    env = game.env
    trial = {}
    trial["illegal_moves"] = 0
    trial["repeat_states"] = 0
    trial["hit_lava"] = False
    trial["reaches_goal"] = False
    trial["actual_path"] = []
    trial["fitness"] = 0

    visited_states = set() 

    for action in trace:
        #time.sleep(0.5)
        _, reward, terminated, truncated, _ = game.step(action, render=False)
        curr_pos = env.agent_pos
        trial["actual_path"].append(action)

        # Let fitness be a function of illegal moves to good moves
        if terminated:
            if reward > 0:
                trial["reaches_goal"] = True
                #trial["fitness"] += env.max_steps # huge reward for hitting goal
                trial["fitness"] += 1000 # huge reward for hitting goal
                break
        elif truncated:
            trial["fitness"] -= env.max_steps # huge negative reward for hitting lava
            break
        else:
            if tuple(curr_pos) in visited_states and action == env.Actions.forward: # Not turning and repeat state
                trial["repeat_states"] += 1
                #print("REPEAAAT STATE")
                trial["fitness"] -= 2 # Small negative reward
            elif action == env.Actions.forward:
                trial["fitness"] += 1 # Reward exploration and making good moves
            elif action == env.Actions.left or action ==env.Actions.right:
                trial["fitness"] -= 0.25 # negative reward for turning (this way if they keep turning its bad)

        visited_states.add(tuple(curr_pos))

    return trial

 
def initialize_population(game: MiniGridEnv, num_population, actions):

    # create the genetic material for the population size and have it be random for each of them other then the first 4, which
    # are DFS traversals with directional preferences
    env = game.env
    
    if num_population < 0:
        print("POP SIZE MUST BE GREATER THAN 0")
        exit()
    
    # create a gene for the population number
    population = [get_first_trace(game, preference_order='u'), 
                  get_first_trace(game, preference_order='d'), 
                  get_first_trace(game, preference_order='u'),
                  get_first_trace(game, preference_order='l')]

    for x in range(num_population - 4):
        rand_trace = []
        # For max steps moves make random moves.
        for i in range(0, env.max_steps):
            rand_trace.append(random.choice(actions))

        # Append the gene
        population.append(rand_trace)

    return population

def genetic_algorithm(game, legal_moves, population, max_generation, probability_crossover, probability_mutation, generational_stats, gen_count):
    winners = set()
    while max_generation > 0:
        fitness = []
        trials = []
        pop_no_extra_moves = []
        for trace in population:
            trial = get_fitness(game, trace)
            fitness.append(trial["fitness"])
            trials.append(trial)
            pop_no_extra_moves.append(trial["actual_path"])
        
        population = pop_no_extra_moves

        # Book Keeping
        final_fitness = fitness.copy()
        final_population = population.copy()
        max_fitness = max(fitness)
        min_fitness = min(fitness)
        avg_fitness = sum(fitness)/len(fitness)
        max_fitness_gene = population[fitness.index(max_fitness)].copy()
        max_fitness_trial = trials[fitness.index(max_fitness)]
        this_generation_stats = (max_fitness, min_fitness, avg_fitness, gen_count)
        generational_stats.append(this_generation_stats)
        max_generation = max_generation - 1
        gen_count = gen_count + 1

        
        good_ones = list(filter(lambda x: x > 900, fitness))
        good_ones1 = list(map(lambda x: tuple(population[fitness.index(x)]), good_ones))

        winners.update(good_ones1)
        print(f"Fitness: {good_ones} MaxFitness {max_fitness} Avg Fitness = {avg_fitness} > {900} = {len(good_ones)} total = {len(winners)}")
        
        # Take our population of ants and get the top 2 for later
        elite1_index, elite2_index = select_max(fitness)
        #print(f"Elite1 {elite1_index, fitness[elite1_index]} Elite2 {elite2_index, fitness[elite2_index]}")

        # Select the two for crossover and mutation (Two Random Rank)
        parent1, parent2 = select(population, fitness)

        # Crossover the selected  with a 2 point crossover (split into thirds, alternate between parents.)
        crossed1, crossed2 = crossover(parent1, parent2, probability_crossover)

        # Grab the information for the elites
        elite1_gene = population[elite1_index].copy()
        elite2_gene = population[elite2_index].copy()

        # Create new population by culling random 2 and replace with the crossed over ones, keep the elite ones in as they were
        cull1_index, cull2_index = cull_random(population, fitness)
        population[cull1_index] = crossed1
        population[cull2_index] = crossed2

        # Mutate the entire population by potentially increasing (wrap around) digits 2 and 3 of every single state. (chance for random reset of 0
        # as well)
        # If elite 2's index changes from removal of 1
        # INSTEAD OF REMOVAL JUST CHANGE WHAT THINGS ARE GETTING SET TO!!!
        population = mutate_population(population, probability_mutation, legal_moves)
       
        # Put back in the the elites
        population[elite1_index] = elite1_gene
        population[elite2_index] = elite2_gene


    return max_fitness, max_fitness_gene, max_fitness_trial, generational_stats, population, final_fitness, final_population, winners

# Perform 3 point crossover based on crossover probability, halfway through
def crossover(parent1, parent2, probability_crossover):

    # If we hit the probability, then crossover on the parents
    random_val = random.random()
    if random_val < probability_crossover:

        p1_cross = int(len(parent1)/5)
        p2_cross = int(len(parent2)/5)

        # Get the poritons for parent 1
        parent1_portion1 = parent1[0:p1_cross]
        parent1_portion2 = parent1[p1_cross:p1_cross * 2]
        parent1_portion3 = parent1[p1_cross * 2:p1_cross * 3]
        parent1_portion4 = parent1[p1_cross * 3:p1_cross * 4]
        parent1_portion5 = parent1[p1_cross * 4:-1]

        # Get the poritons for parent 2
        parent2_portion1 = parent2[0:p2_cross]
        parent2_portion2 = parent2[p2_cross:p2_cross * 2]
        parent2_portion3 = parent2[p2_cross * 2:p2_cross * 3]
        parent2_portion4 = parent2[p2_cross * 3:p2_cross * 4]
        parent2_portion5 = parent2[p2_cross * 4:-1]

        
        # Perform the crossover
        crossover1 = parent1_portion1 + parent2_portion2 + parent1_portion3 + parent2_portion4 + parent1_portion5
        crossover2 = parent2_portion1 + parent1_portion2 + parent2_portion3 + parent1_portion4 + parent2_portion5

        # Return the Crossover
        return crossover1, crossover2
    else:
        return parent1, parent2

# Mutate the entire population by potentially adding 2 moves in the mix randomly
def mutate_population(population, probability_mutation, legal_moves):
    for trace in population:
        random_val = random.random()

        # If we hit the probabilty, mutate
        if random_val < probability_mutation:

            #Get transitions out of state1 and state2 + new random checks
            rand_move1 = random.choice(legal_moves)
            rand_move2 = random.choice(legal_moves)

            rand_index1 = random.randint(0, len(trace))
            rand_index2 = random.randint(0, len(trace))
            trace.insert(rand_index1, rand_move1)
            trace.insert(rand_index2, rand_move2)
               
    return population
       
# Select and return two via roulette, with probability of choice weighted on their fitness in the old generation
def select(old_gen, fitness):
    old_gen1 = old_gen.copy()
    fitness1 = fitness.copy()

    # Get the first gene
    # Normalize the weights
    min_value = min(fitness1)
    fitness2 = list(map(lambda x: (x + abs(min_value) + 1), fitness1))

    gene1 = random.choices(old_gen1, weights=fitness2, k=1)
    gene1 = gene1[0]

    # Get the index from first gene and  remove the fitness of it from fitness, and the gene from oldgen to get the second
    gene1_index = old_gen1.index(gene1)
    del(old_gen1[gene1_index])
    del(fitness2[gene1_index])

    # Get the second
    #print(f"Fitness {fitness}")
    gene2 = random.choices(old_gen1, weights=fitness2, k=1)
    gene2 = gene2[0] 

    return gene1, gene2

# Select and return two best indexes in the old generation for elitism later. Remember their indexes
def select_max(fitness):
    fitness1 = fitness.copy()

    # Get the first index
    max1 = (max(fitness1))
    index1 = fitness.index(max1) 

    # Remove first by setting it to 0 and get second best index
    del(fitness1[index1])
    max2 = (max(fitness1))
    index2 = fitness.index(max2) 

    return index1, index2

# Cull random 2 in population (keeping the elite values) and then add in the mutated. This produces the next generation.
def cull_random(population, fitness):


    # Inverse the fitness so we can weigh bad ones higher, +1 for smoothing of weights
    inverse = fitness.copy()
    population1 = population.copy()
    max_inverse = max(inverse)
    inverse = list(map(lambda x: max_inverse - x + 1, inverse))

    # Cull two random ones weighted towards the bad ones.
    to_cull1 = random.choices(population1, weights=inverse, k = 1)
    to_cull1 = to_cull1[0]
    to_cull1_index = population.index(to_cull1)
    del(inverse[to_cull1_index])
    del(population1[to_cull1_index])

    to_cull2 = random.choices(population1, weights=inverse, k = 1)
    to_cull2 = to_cull2[0]
    to_cull2_index = population.index(to_cull2)

    # Return Indexes to replace
    return to_cull1_index, to_cull2_index

# DFS Traversal to get a winning path.
def get_first_trace(game, preference_order='u'): # preference order is what node to explore first.
    # Get to key
    game.start(render=False)

    # Get to the key
    path_to_key, key_location = dfs(game, preference_order, goal='key', next_to_goal=True)
    key_actions = get_actions(path_to_key)
    to_key = path_to_steps(game, key_actions)
    print(key_actions, key_location)
    
    # Pickup Key
    pickup_key = do_objective(game, key_location, game.env.Actions.pickup)

    # Get to door
    path_to_door, door_location = dfs(game, preference_order, goal='door', next_to_goal=True)
    door_actions = get_actions(path_to_door) 
    to_door = path_to_steps(game, door_actions)

    # Unlock Door
    unlock_door = do_objective(game, door_location, game.env.Actions.toggle)

    # Get to Box
    path_to_box, box_location = dfs(game, preference_order, goal='box', next_to_goal=True)
    box_actions = get_actions(path_to_box)
    to_box = path_to_steps(game, box_actions)

    # Face the box 
    face_box = do_objective(game, box_location, None)

    # Drop the key
    move_choice = random.choice([game.env.Actions.right, game.env.Actions.left]) # To not overload with one direction
    drop_key = [move_choice, move_choice, game.env.Actions.drop] # Drop behind us 

    # Pickup the Box
    move_choice = random.choice([game.env.Actions.right, game.env.Actions.left]) # To not overload with one direction
    pickup_box = [move_choice, move_choice, game.env.Actions.pickup] # Turn and get the last goal

    total_actions = to_key + pickup_key + to_door + unlock_door + to_box + face_box + drop_key + pickup_box

    return total_actions

# Steps to pickup an object
def do_objective(game, goal_direction, objective):
    game_steps = []
    moves = objective_moves(game, goal_direction, objective)

    for move in moves:
        game_steps.append(move)
        game.step(move,render=False)

    return game_steps

# Steps to move around
def path_to_steps(game, actions):
    game_steps = []
    for action in actions:
        moves = objective_moves(game, action, game.env.Actions.forward)
        
        for move in moves:
            game_steps.append(move)
            game.step(move, render=False)

    return game_steps

# Turn a directional move into a good valid step of turns and objective
def objective_moves(game, goal_direction, objective):
    env = game.env
    direction = env.agent_dir #R0 D1 L2 U3
    # Do Turning
    # If have to turn left once
    if direction == 0 and goal_direction == 'u' or \
        direction == 1 and goal_direction == 'r' or \
        direction == 2 and goal_direction == 'd' or \
        direction == 3 and goal_direction == 'l':
        moves = [game.env.Actions.left]

    # If have to turn right once
    elif direction == 0 and goal_direction == 'd' or \
        direction == 1 and goal_direction == 'l' or \
        direction == 2 and goal_direction == 'u' or \
        direction == 3 and goal_direction == 'r':
        moves = [game.env.Actions.right]

    # If have to turn behind us
    elif direction == 0 and goal_direction == 'l' or \
        direction == 1 and goal_direction == 'u' or \
        direction == 2 and goal_direction == 'r' or \
        direction == 3 and goal_direction == 'd':
        move_choice = random.choice([game.env.Actions.right, game.env.Actions.left]) # To not overload with one direction
        moves = [move_choice, move_choice]
    else:
        moves = []

    # Do action
    if objective:
        moves = moves + [objective]

    return moves

def dfs(game, preference_order, goal='goal', next_to_goal=False):
    env = game.env
    start = env.agent_pos
    grid = env.grid
    stack = [(start, 'start', None)]
    visited = [(start, 'start', None)]
    path = []
    while stack:
        curr = stack.pop()
        curr_pos = curr[0]
        path.append(curr)
        neighbors = []
        left = curr_pos + np.array([-1, 0])
        right = curr_pos + np.array([1, 0])
        up = curr_pos + np.array([0, -1])
        down = curr_pos + np.array([0, 1])

        if preference_order == 'u':
            neighbors.append((up, 'u', curr)) #up
            neighbors.append((down, 'd', curr)) #down
            neighbors.append((left, 'l', curr)) #left
            neighbors.append((right, 'r', curr)) #right
        elif preference_order == 'd':
            neighbors.append((down, 'd', curr)) #down
            neighbors.append((left, 'l', curr)) #left
            neighbors.append((right, 'r', curr)) #right
            neighbors.append((up, 'u', curr)) #up
        elif preference_order == 'l':
            neighbors.append((left, 'l', curr)) #left
            neighbors.append((right, 'r', curr)) #right
            neighbors.append((up, 'u', curr)) #up
            neighbors.append((down, 'd', curr)) #down
        elif preference_order == 'r':
            neighbors.append((right, 'r', curr)) #right
            neighbors.append((up, 'u', curr)) #up
            neighbors.append((down, 'd', curr)) #down
            neighbors.append((left, 'l', curr)) #left
        for (neighbor, action, parent) in neighbors:
            neighbor = list(neighbor)
            if valid_move(game, neighbor) and neighbor not in visited:
                visited.append(neighbor)
                stack.append((neighbor, action, parent))
                reached_goal = is_goal(game, neighbor, goal=goal, next_to_goal=next_to_goal)

                if reached_goal:
                    path.append((neighbor,action, parent))

                    if next_to_goal:
                        return backtrack_path(path), reached_goal
                    else:
                        return backtrack_path(path)
        
# Check if position is a valid move
def valid_move(game, neighbor):
    env = game.env
    grid = env.grid
    tile = grid.get(*neighbor)
    
    # If not null it is something
    if tile:
        if tile.type == 'wall':
            return False
        elif tile.type == 'box':
            return False
        elif tile.type == 'door' and tile.is_locked == False:
            return True
        elif tile.type == 'key':
            return False
        else:
            return False

    return True

# Check if position is a goal position
def is_goal(game, position, goal='goal', next_to_goal=False):
    env = game.env
    grid = env.grid
    tile = grid.get(*position)
    
    # If we are checking if we are next to a goal
    if next_to_goal:
        position = np.array(position)
        # Above
        up_position = position + np.array([0, -1])
        up_goal = is_goal(game, up_position, goal=goal, next_to_goal=False)

        # Below
        down_position = position + np.array([0, 1])
        down_goal = is_goal(game, down_position, goal=goal, next_to_goal=False)

        # Right
        right_position = position + np.array([1, 0])
        right_goal = is_goal(game, right_position, goal=goal, next_to_goal=False)

        # Left
        left_position = position + np.array([-1, 0])
        left_goal = is_goal(game, left_position, goal=goal, next_to_goal=False)

        if up_goal:
            return 'u'
        elif down_goal:
            return 'd'
        elif left_goal:
            return 'l'
        elif right_goal:
            return 'r'
        else:
            return None

    if tile:
        if tile.type == goal:
            return True
    else:
        return False

def backtrack_path(path):
    proper_path =[]
    curr = path[-1]
    while curr[2] != None:
        proper_path.append(curr)
        curr = curr[2]
    
    proper_path.reverse()
    return proper_path

def get_actions(path):
    actions = []
    for (_, action, _) in path:
        actions.append(action)
    
    return actions
  