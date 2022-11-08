import MapUtils as mp
from random import choice




def basic_update_agent(agent, predator, prey, estimated_predator_position=None, estimated_prey_position=None):
    neighbors_and_self = agent.graph[agent.position][:]
    neighbors_and_self.append(agent.position)

    if estimated_predator_position is None:
        predator_distances = mp.getShortestDistancesToGoals(agent.graph, predator.position, neighbors_and_self[:])
    else:
        predator_distances = mp.getShortestDistancesToGoals(agent.graph, estimated_predator_position, neighbors_and_self[:])

    if estimated_prey_position is None:
        prey_distances = mp.getShortestDistancesToGoals(agent.graph, prey.position, neighbors_and_self[:])
    else:
        prey_distances = mp.getShortestDistancesToGoals(agent.graph, estimated_prey_position, neighbors_and_self[:])

    # print(predator_distances)
    # print(prey_distances)
    cur_dist_pred = predator_distances[agent.position]
    cur_dist_prey = prey_distances[agent.position]
    smallest_prey = agent.config["GRAPH_SIZE"] + 1
    smallest_prey_pos = -1
    largest_pred = -1
    largest_pred_pos = -1

    for position in predator_distances.keys():

        if prey_distances[position] <= smallest_prey:
            smallest_prey = prey_distances[position]
            smallest_prey_pos = position
        if predator_distances[position] >= largest_pred:
            largest_pred = predator_distances[position]
            largest_pred_pos = position

    # print(smallest_prey, predator_distances[smallest_prey_pos], cur_dist_prey, cur_dist_pred)
    # print(prey_distances[largest_pred_pos], largest_pred, cur_dist_prey, cur_dist_pred)

    closer_to_prey = set([x for x in prey_distances.keys() if prey_distances[x] < cur_dist_prey and x != agent.position] )
    same_to_prey = set([x for x in prey_distances.keys() if prey_distances[x] == cur_dist_prey and x != agent.position])
    far_from_prey = set([x for x in prey_distances.keys() if prey_distances[x] > cur_dist_prey and x != agent.position])
    closer_to_pred = set([x for x in predator_distances.keys() if predator_distances[x] < cur_dist_pred and x != agent.position])
    same_to_pred = set([x for x in predator_distances.keys() if predator_distances[x] == cur_dist_pred and x != agent.position])
    far_from_pred = set([x for x in predator_distances.keys() if predator_distances[x] > cur_dist_pred and x != agent.position])

    closer_to_prey_and_further_from_pred = closer_to_prey.intersection(far_from_pred)
    closer_to_prey_and_same_from_pred = closer_to_prey.intersection(same_to_pred)
    same_to_prey_and_further_from_pred = same_to_prey.intersection(far_from_pred)
    same_to_prey_and_same_from_pred = same_to_prey.intersection(same_to_pred)


    if len(closer_to_prey_and_further_from_pred) != 0:
        agent.position = choice(list(closer_to_prey_and_further_from_pred))
    elif len(closer_to_prey_and_same_from_pred) != 0:
        agent.position = choice(list(closer_to_prey_and_same_from_pred))
    elif len(same_to_prey_and_further_from_pred) != 0:
        agent.position = choice(list(same_to_prey_and_further_from_pred))
    elif len(same_to_prey_and_same_from_pred) != 0:
        agent.position = choice(list(same_to_prey_and_same_from_pred))
    elif len(far_from_pred) != 0:
        agent.position = choice(list(far_from_pred))
    elif len(same_to_pred) != 0:
        agent.position = choice(list(same_to_pred))

    return 1 if agent.position == prey.position else -1 if agent.position == predator.position else 0
