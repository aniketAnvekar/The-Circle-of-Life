import Painter as p
import Graph as gr
import Prey as P
import Predator as Pr
import EasyPredator as EPr
import json
import random
import AgentUtils as au
import Agent1 as A1
import Agent2 as A2
import Agent3 as A3
import Agent4 as A4
import Agent5 as A5
import Agent6 as A6
import Agent7 as A7
import Agent7B as A7B
import Agent7C as A7C
import Agent8 as A8
import Agent8B as A8B
import Agent8C as A8C
import Agent10A as A10A
import Agent10B as A10B



def load_config():
    with open("./config.json", "r") as f:
        return json.load(f)


def init():
    # load run time details
    config = load_config()

    # create graph
    graph = gr.Graph(config["GRAPH_SIZE"], config)
    graph.create()

    # setting up graphics
    frame = p.init_frame()
    networkx_graph = graph.visualize()
    fig, axis, node_positions, xlim, ylim = p.set_up_figure(networkx_graph)
    canvas = p.init_canvas(frame, fig)

    # setting up players
    agent_start = random.randrange(0, config["GRAPH_SIZE"])
    agent = A10A.Agent10A(graph.alist, agent_start, config)
    predator = Pr.Predator(graph.alist, config, agent_start)
    prey = P.Prey(graph.alist, config, agent_start)

    # setting update function
    frame.after(config["TIME_DELAY"],
                lambda: update(frame, canvas, config, networkx_graph, fig, axis, node_positions, xlim, ylim, predator,
                               prey, agent))

    return frame


def update(frame, canvas, config, networkx_graph, fig, axis, node_positions, xlim, ylim, predator, prey, agent):
    # update players
    # shortest distances from predator to neighbors
    # shortest distances from agent to prey

    status = agent.update(predator, prey)
    if status == 1:
        print("Agent Win...")
        return
    elif status == -1:
        print("Predator Win...")
        return

    status = prey.update(agent.position)
    if status == 1:
        print("Agent Win...")
        return
    elif status == -1:
        print("Predator Win...")
        return
    status = predator.update(agent.position)
    if status == 1:
        print("Agent Win...")
        return
    elif status == -1:
        print("Predator Win...")
        return
    au.belief_system_move_pieces(agent)
    # input()

    # redraw the graph
    p.draw_next_graph(canvas, networkx_graph, axis, node_positions, xlim, ylim, predator, prey, agent)

    # reset update function
    frame.after(config["TIME_DELAY"],
                lambda: update(frame, canvas, config, networkx_graph, fig, axis, node_positions, xlim, ylim, predator,
                               prey, agent))


def free_run():
    frame = init()
    frame.mainloop()


def trials():
    config = load_config()
    graph = gr.Graph(config["GRAPH_SIZE"], config)
    graph.create()
    agent_start = random.randrange(0, config["GRAPH_SIZE"])
    agent = A10A.Agent10A(graph.alist, agent_start, config)
    predator = Pr.Predator(graph.alist, config, agent_start)
    prey = P.Prey(graph.alist, config, agent_start)
    timeouts = 0
    deaths = 0
    success = 0
    total_prey_guess = 0
    total_pred_guess = 0
    total_prey_correct = 0
    total_pred_correct = 0
    for i in range(config["NUMBER_OF_GRAPHS"]):
        print("GRAPH " + str(i))
        for k in range(config["TRIALS_PER_GRAPH"]):
            break_flag = False
            status = 0
            for j in range(config["TIMEOUT"]):
                status = agent.update(predator, prey)
                if status != 0:
                    break_flag = True
                    break

                status = prey.update(agent.position)
                if status != 0:
                    break_flag = True
                    break
                status = predator.update(agent.position)
                if status != 0:
                    break_flag = True
                    break

                au.belief_system_move_pieces(agent)

            if break_flag:
                if status == 1:
                    success = success + 1
                else:
                    deaths = deaths + 1
            else:
                timeouts = timeouts + 1

            total_prey_guess = agent.total_prey_guess + total_prey_guess
            total_pred_guess = agent.total_pred_guess + total_pred_guess
            total_prey_correct = agent.total_prey_correct + total_prey_correct
            total_pred_correct = agent.total_pred_correct + total_pred_correct
            agent_start = random.randrange(0, config["GRAPH_SIZE"])
            agent = A3.Agent3(graph.alist, agent_start, config)
            predator = Pr.Predator(graph.alist, config, agent_start)
            prey = P.Prey(graph.alist, config, agent_start)

        graph = gr.Graph(config["GRAPH_SIZE"], config)
        graph.create()

    print("Timeouts: " + str(timeouts) + ", " + str(timeouts/(timeouts + deaths + success)))
    print("Deaths: " + str(deaths) + ", " + str(deaths/(timeouts + deaths + success)))
    print("Wins: " + str(success) + ", " + str(success/(timeouts + deaths + success)))
    if total_prey_guess != 0:
        print("Prey Survey Rate: " + str(total_prey_correct) + " correct of " + str(total_prey_guess) + ", " + str(total_prey_correct / total_prey_guess))
    if total_pred_guess != 0:
        print("Predator Survey Rate: " + str(total_pred_correct) + " correct of " + str(total_pred_guess) + ", " + str(total_pred_correct / total_pred_guess))


trials()
