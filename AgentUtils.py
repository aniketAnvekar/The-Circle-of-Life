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


    def gen_belief_agent_update_found_case(agent, transition_matrix):
        agent.q = list(np.dot(transition_matrix, agent.q))
        agent.checkProbSum(sum(agent.q))

        old_agent_pos_prob = agent.q[agent.position]
        agent.q = list(map(lambda x: x / (1 - old_agent_pos_prob), agent.q))
        agent.q[agent.position] = 0

    def prey_belief_agent_update(agent, prey):
        if not agent.found_prey:
            # initialization
			agent.q = [1/(agent.config["GRAPH_SIZE"] - 1) for i in range(agent.config["GRAPH_SIZE"])]
			agent.q[agent.position] = 0
        else:
            gen_belief_agent_update_found_case(agent, agent.P)

    def predator_belief_agent_update(agent, predator):

		if not agent.found_predator: # agent does know where the predator starts
			agent.q = [0 for i in range(agent.config["GRAPH_SIZE"])]
			agent.q[predator.position] = 1
		else:
            P = agent.calculate_transition_probability_matrix()
			gen_belief_agent_update_found_case(agent, P)

    def gen_belief_survey_update(agent, found, survey_spot):
        if found:
            agent.q = [0 for i in range(agent.config["GRAPH_SIZE"])]
			agent.q[survey_spot] = 1
        else:
            old_survey_spot_prob = agent.q[survey_spot]
			agent.q[survey_spot] = 0
			agent.q = list(map(lambda x: x / (1 - old_survey_spot_prob), agent.q))

    def prey_belief_survey_update(agent, prey):
        max_prob = max(agent.q)
		survey_spot = choice([i for i in agent.graph.keys() if agent.q[i] == max_prob])
        agent.found_prey = (survey_spot == prey.position) or agent.found_prey

        gen_belief_survey_update(agent, survey_spot == prey.position, survey_spot)
		agent.checkProbSum(sum(agent.q))

		max_prob = max(agent.q)
		return choice([i for i in agent.graph.keys() if agent.q[i] == max_prob])


    def predator_belief_survey_update(agent, predator):
        agent.checkProbSum(sum(agent.q))

		max_prob = max(agent.q)
		survey_spot = choice([i for i in agent.graph.keys() if agent.q[i] == max_prob])
        agent.found_predator = (survey_spot == predator.position) or agent.found_predator

        gen_belief_survey_update(agent, survey_spot == predator.position, survey_spot)
		agent.checkProbSum(sum(agent.q))

        max_prob = max(agent.q)
		return choice([i for i in agent.graph.keys() if agent.q[i] == max_prob])
