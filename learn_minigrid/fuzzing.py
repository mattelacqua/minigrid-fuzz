from copy import copy, deepcopy
import random
import numpy as np
from minigrid.minigrid_env import MiniGridEnv
import time




def fuzz(unwrapped_env, env, search_trace, population_size=100, num_generations=100, elite_cull_ratio=0.10, probability_crossover=0.3, probability_mutation=0.2):
    # Randomly initialize population
    env.reset()
    starting_position = env.agent_pos
    key_steps, door_steps, box_steps, key_start_dir, door_start_dir, box_start_dir = search_trace

    # For the keys
    key_pop = initialize_pop(env, [key_steps], population_size) 
    success_key_traces = genetic_algorithm(unwrapped_env, env, key_pop, \
                            population_size=10, \
                            num_generations=num_generations, \
                            elite_cull_ratio=elite_cull_ratio, \
                            probability_crossover=probability_crossover, \
                            probability_mutation=probability_mutation, \
                            key_traces=None, door_traces=None)


    # For the doors
    door_pop = initialize_pop(env, [door_steps], population_size) 
    success_door_traces = genetic_algorithm(unwrapped_env, env, door_pop, \
                            population_size=10, \
                            num_generations=num_generations, \
                            elite_cull_ratio=elite_cull_ratio, \
                            probability_crossover=probability_crossover, \
                            probability_mutation=probability_mutation, \
                            key_traces=success_key_traces, door_traces=None)

    # For the boxes
    box_pop = initialize_pop(env, [box_steps], population_size) 
    success_box_traces = genetic_algorithm(unwrapped_env, env, box_pop,\
                            population_size=10, \
                            num_generations=num_generations, \
                            elite_cull_ratio=elite_cull_ratio, \
                            probability_crossover=probability_crossover, \
                            probability_mutation=probability_mutation, \
                            key_traces=success_key_traces, door_traces=success_door_traces)
    
    # Agregate the best combinations of them.
    # include the translations. Assume we move onto the position we do the action on as well.
    print(f"Good keys {len(success_key_traces)}")
    print(f"Good door {len(success_door_traces)}")
    print(f"Good box {len(success_box_traces)}")

    precursor_steps = [min(success_key_traces, key=len), min(success_door_traces, key=len), min(success_box_traces, key=len)]
    #env.render()
    unwrapped_env.env.render_mode = 'human'
    
    for trace in success_key_traces:
        env.reset()
        print(f"Starting Dir {env.agent_dir}")
        for step in trace:
            time.sleep(0.7)
            print(f"{step.name}")
            env.step(step)
        print(f"Ending Dir {env.agent_dir}")
        time.sleep(2)

    while True:
        env.reset()
        for step_set in precursor_steps:
            for step in step_set:
                print(f"Agent Direction {env.agent_dir} Is moving {step.name}")
                time.sleep(1.5)
                env.step(step)

    return 


def initialize_pop(env, population, population_size):
    grid_size = env.room_size * env.room_size
    for trace in range(0, population_size-1):
        trace = []
        for i in range(0, grid_size):
            trace.append(random.choice(['u', 'd', 'l', 'r']))
        population.append(trace)
    return population.copy()


def genetic_algorithm(unwrapped_env, env, population, \
                        population_size=100, \
                        num_generations=100, \
                        elite_cull_ratio=0.10, \
                        probability_crossover=0.3, \
                        probability_mutation=0.2,
                        key_traces=None, door_traces=None):
    
    # check what portion of task we are fuzzing
    goal = None 
    if not key_traces and not door_traces:
        goal = 'key'
    elif key_traces and not door_traces:
        goal = 'door'
    else:
        goal = 'box'

    # Set parameters
    elite_cull_count = int(population_size*elite_cull_ratio)
    gen_count = 0
    generational_stats = []
    winners = set()
    precursor_steps = []

    while gen_count < num_generations:
        print(f"Generation {gen_count}")
        fitness = []
        action_paths = []
        trials = []
        init_directions = []
        pop_no_extra_moves = []
        for trace in population:
            # Choose a random successful precursory steps
            if goal == 'door':
                precursor_steps = [min(key_traces, key=len)]
            if goal == 'box':
                precursor_steps = [min(key_traces, key=len), min(door_traces, key=len)]
            trial = get_fitness(env, trace, goal, precursor_steps)
            fitness.append(trial["fitness"])
            trials.append(trial.copy())
            #init_directions.append(trial['init_direction'])
            pop_no_extra_moves.append(trial["actual_path"].copy())
            action_paths.append(trial["action_path"].copy())
        
        population = pop_no_extra_moves

        # Book Keeping
        final_fitness = fitness.copy()
        final_population = population.copy()
        max_fitness = max(fitness)
        min_fitness = min(fitness)
        avg_fitness = sum(fitness)/len(fitness)
        max_fitness_individual = population[fitness.index(max_fitness)].copy()
        this_generation_stats = (max_fitness, min_fitness, avg_fitness, gen_count)
        generational_stats.append(this_generation_stats)

        
        # Reward is for reaching the goal. Will get 100. Minus 2 for repeat states, so any positive value will have reached goal.
        optimal_value = env.max_steps - (env.room_size * env.room_size)
        print(f"max steps {env.max_steps} room_size {env.room_size}, optimal_value {optimal_value}")
        successful_traces_fit = list(filter(lambda x: x > optimal_value, fitness))
        for successful_fitness in successful_traces_fit:
            successful_trace = action_paths[fitness.index(successful_fitness)]
            winner = tuple(successful_trace.copy())
            winners.add(winner)

        print(f"Fitness: {fitness} MaxFitness {max_fitness} Avg Fitness = {avg_fitness} > {0} = {len(successful_traces_fit)} total = {len(winners)}")
        
        # Take our population of individual and get the top n% for later
        elites, elite_indexes = select_max(population, fitness, elite_cull_count)

        # Select the two for crossover and mutation (Two Random Rank)
        parent1, parent2 = select(population, fitness)

        # Crossover the selected  with a random single point crossover 
        crossed1, crossed2 = crossover(parent1, parent2, probability_crossover)

        # Mutate the entire population by potentially increasing and decreasing (wrap around) 2 genes
        population = mutate_population(env, population, probability_mutation)

        # Create new population by culling bottom 20% and replacing with the elite ones and crossed ones
        cull_random(population, fitness, action_paths, elite_cull_count, elites, crossed1, crossed2)

        gen_count += 1


    return list(winners)
def get_fitness(env, trace, goal, precursor_steps):
    #time.sleep(3)
    
    # Convert the 'u' 'd' 'l' 'r' to turn and forward actions

    objective = env.Actions.pickup
    if goal == 'door':
        objective = env.Actions.toggle

    trial = {}
    trial["illegal_moves"] = 0
    trial["repeat_states"] = 0
    trial["hit_lava"] = False
    trial["reaches_goal"] = False
    trial["action_path"] = []
    trial["actual_path"] = []
    trial["fitness"] = 0

    visited_states = set() 

    env.reset()
    print(f"fitness starting dir {env.agent_dir}")
    for step_set in precursor_steps:
        # do the precursor steps
        for step in step_set:
            env.step(step)

    for action in trace:

        # add to actual path
        trial['actual_path'].append(action)
        step_actions = objective_moves(env, action, env.Actions.forward)
        
        # Get our current position
        curr_pos = env.agent_pos
        visited_states.add(tuple(curr_pos))

        # do the actions of the directional udlr step
        for move in step_actions:
            trial['action_path'].append(move)
            #time.sleep(0.05)
            _, _, _, truncated, _ = env.step(move)
            
            # Get our current position
            curr_pos = env.agent_pos
            
            # If took max steps, end and reward as bad
            if truncated:
                trial['fitness'] -= 1000
                return trial

            # Reward shaping ( if repeat action)
            if tuple(curr_pos) in visited_states and action == env.Actions.forward: # Not turning and repeat state
                trial["repeat_states"] += 1
                trial["fitness"] -= 2 # Small negative reward
        
        # if we are at a goal, add actions to pickup/unlock and then step on the tile.
        found_goal = is_goal(env, curr_pos, goal=goal, next_to_goal=True)
        
        # If we hit the goal
        if found_goal:
            trial['reaches_goal'] = True
            trial['fitness'] += (env.max_steps-len(trial['actual_path']))
            objective_steps = do_objective(env, found_goal, objective)
            
            # If its a door we have to unlock, drop, and then step on the door tile.
            if goal == 'door':
                if len(objective_steps) > 1:
                    for action in objective_steps[:-1]:
                        if action == env.Actions.left:
                            objective_steps.append(env.Actions.left)
                            objective_steps.append(env.Actions.drop)
                            objective_steps.append(env.Actions.right)
                        elif action == env.Actions.right:
                            objective_steps.append(env.Actions.right)
                            objective_steps.append(env.Actions.drop)
                            objective_steps.append(env.Actions.left)

                # If its ahead of us, turn all the way around, drop it, turn back to where we were
                else:
                    action = random.choice([env.Actions.right, env.Actions.left])
                    objective_steps.extend([action, action, env.Actions.drop, action, action])

            # perform objective:
            for step in objective_steps:
                trial['action_path'].append(step)
                #env.step(step)

            # Step onto the goal tile
            env.step(env.Actions.forward)
            trial['action_path'].append(env.Actions.forward)
            
            # Adjust to face Right
            if env.agent_dir == 1: # if down, left once
                env.step(env.Actions.left)
                trial['action_path'].extend([env.Actions.left])
            if env.agent_dir == 2: # if left, left twice
                trial['action_path'].extend([env.Actions.left, env.Actions.left])
                env.step(env.Actions.left)
                env.step(env.Actions.left)
            if env.agent_dir == 3: # if up, right once
                trial['action_path'].extend([env.Actions.right])
                env.step(env.Actions.right)

            if env.agent_dir != 0:
                print(f"Fitness Ending dir {env.agent_dir}")
            
            return trial

    return trial
# Perform singlepoint crossover based on crossover probability
def crossover(parent1, parent2, probability_crossover):

    # If we hit the probability, then crossover on the parents
    random_val = random.random()
    if random_val < probability_crossover and (len(parent1) > 2 and len(parent2) > 2):

        # pick a random division for the crossover
        cross_point = random.randrange(1, min(len(parent1), len(parent2)) - 1)

        # Perform Crossover
        crossover1 = parent1[:cross_point] + parent2[cross_point:]
        crossover2 = parent2[:cross_point] + parent1[cross_point:]
        
        # Return the Crossover
        return crossover1, crossover2

    else:
        return parent1, parent2

# Mutate the entire population by potentially adding 2 moves in the mix randomly
def mutate_population(env, population, probability_mutation):
    for trace in population:
        random_val = random.random()

        # If we hit the probabilty, mutate
        if random_val < probability_mutation:

            #Get transitions out of state1 and state2 + new random checks
            rand_move1 = random.choice(['u', 'd', 'l', 'r'])
            rand_move2 = random.choice(['u', 'd', 'l', 'r'])

            rand_index1 = random.randint(0, len(trace))
            rand_index2 = random.randint(0, len(trace))
            trace.insert(rand_index1, rand_move1)
            trace.insert(rand_index2, rand_move2)
               
    return population
       
# Select and return two via roulette, with probability of choice
# weighted on their fitness in the old generation
def select(old_gen, fitness):
    old_gen_temp = old_gen.copy()
    fitness_temp = fitness.copy()

    # Get the first individual
    # Normalize the weights
    min_value = min(fitness_temp)
    fitness_temp = list(map(lambda x: (x + abs(min_value) + 1), fitness_temp))

    individual1 = random.choices(old_gen_temp, weights=fitness_temp, k=1)
    individual1 = individual1[0]

    # Get the index from first individual and  remove the fitness of it from fitness, 
    # and the individual from oldgen to get the second
    individual1_index = old_gen_temp.index(individual1)
    del(old_gen_temp[individual1_index])
    del(fitness_temp[individual1_index])

    # Get the second individual
    individual2 = random.choices(old_gen_temp, weights=fitness_temp, k=1)
    individual2 = individual2[0] 

    return individual1, individual2

# Select and return n best individuals in the old generation
def select_max(population, fitness, n):
    fitness_temp = fitness.copy()

    elites = []
    elite_indexes = []
    for i in range(0, n):
        # Get the index of the max fitness
        elite_fitness = max(fitness_temp)
        elite_index = fitness.index(elite_fitness) 
        elite_temp_index = fitness_temp.index(elite_fitness) 
        elite_individual = population[elite_index].copy()

        # Add to our list of elites and elite indices
        elites.append(elite_individual)
        elite_indexes.append(elite_index)

        # Remove from temp to keep getting next max
        del(fitness_temp[elite_temp_index])

    return elites, elite_indexes

# Cull random n in population, this produces the next generation.
def cull_random(population, fitness, action_paths, n, elites, crossed1, crossed2):

    # Inverse the fitness so we can weigh bad ones higher, +1 for smoothing of weights
    inverse = fitness.copy()
    max_inverse = max(inverse)
    inverse = list(map(lambda x: max_inverse - x + 1, inverse))

    # Cull two random ones weighted towards the bad ones.
    for individual in range(0,n+2):
        culled = random.choices(population, weights=inverse, k = 1)
        culled = culled[0]
        culled_index = population.index(culled)
        del(inverse[culled_index])
        del(population[culled_index])
        del(action_paths[culled_index])
        
    
    # reintroduce crossovers / elites
    population.extend(elites)
    population.extend([crossed1, crossed2])

def search(env: MiniGridEnv, env_unwrapped):

    # create the genetic material for the population size and have it be random for each of them other then the first 4, which
    env.reset()
    
    # create a gene for the population number
    #env.render()
    trace = get_first_traces(env, preference_order='u')

    return trace
# DFS Traversal to get a winning path.
def get_first_traces(env, preference_order='u'): # preference order is what node to explore first.
    # Get to the key
    key_start_dir = env.agent_dir
    path_to_key, key_location = dfs(env, preference_order, goal='key', next_to_goal=True)
    key_actions = get_actions(path_to_key)
    to_key = path_to_steps(env, key_actions)
    #print(key_actions, key_location)
    
    # Pickup Key
    pickup_key = do_objective(env, key_location, env.Actions.pickup)

    # step into where key was
    env.step(env.Actions.forward)
    door_start_dir = env.agent_dir

    # Get to door
    path_to_door, door_location = dfs(env, preference_order, goal='door', next_to_goal=True)
    door_actions = get_actions(path_to_door) 
    to_door = path_to_steps(env, door_actions)

    # Unlock Door
    unlock_door = do_objective(env, door_location, env.Actions.toggle)

    # Drop the key

    drop_key = []
    # Drop it in the place we just came from
    if len(unlock_door) > 1:
        for action in unlock_door[:-1]:
            if action == env.Actions.left:
                drop_key.append(env.Actions.left)
                drop_key.append(env.Actions.drop)
                drop_key.append(env.Actions.right)
            elif action == env.Actions.right:
                drop_key.append(env.Actions.right)
                drop_key.append(env.Actions.drop)
                drop_key.append(env.Actions.left)

    # If its ahead of us, turn all the way around
    else:
        action = random.choice([env.Actions.right, env.Actions.left])
        drop_key.extend([action, action, env.Actions.drop, action, action])
    
    # step into door
    env.step(env.Actions.forward)

    #print(f"Drop steps{drop_key}")
    # Get to Box
    box_start_dir = env.agent_dir
    path_to_box, box_location = dfs(env, preference_order, goal='box', next_to_goal=True)
    box_actions = get_actions(path_to_box)
    to_box = path_to_steps(env, box_actions)

    # Face & pickup the box 
    pickup_box = do_objective(env, box_location, env.Actions.pickup)

    #total_actions = to_key + pickup_key + to_door + unlock_door + drop_key + to_box + pickup_box

    print(f"key_actions: {key_actions}")
    print(f"door_actions: {door_actions}")
    print(f"box_actions: {box_actions}")

    return (key_actions, door_actions, box_actions, key_start_dir, door_start_dir, box_start_dir)

# Steps to pickup an object
def do_objective(game, goal_direction, objective):
    game_steps = []
    moves = objective_moves(game, goal_direction, objective)

    for move in moves:
        game_steps.append(move)
        game.step(move)

    return game_steps

# Steps to move around
def path_to_steps(game, actions):
    game_steps = []
    for action in actions:
        moves = objective_moves(game, action, game.env.Actions.forward)
        
        for move in moves:
            game_steps.append(move)
            game.step(move)

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
  