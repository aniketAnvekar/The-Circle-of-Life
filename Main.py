import Painter as p
import Graph as gr
import Prey as P
import Predator as Pr
import EasyPredator as EPr
import json
import random
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
    agent = A2.Agent2(graph.alist, agent_start, config)
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

    input()

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
    agent = A1.Agent1(graph.alist, agent_start, config)
    predator = Pr.Predator(graph.alist, config, agent_start)
    prey = P.Prey(graph.alist, config, agent_start)
    timeouts = 0
    deaths = 0
    success = 0
    for i in range(config["NUMBER_OF_TRIALS"]):
        if i % 50 == 0:
            print("TRIAL " + str(i))
        breakFlag = False
        status = 0
        for j in range(config["TIMEOUT"]):
            status = agent.update(predator, prey)
            if status != 0:
                breakFlag = True
                break

            status = prey.update(agent.position)
            if status != 0:
                breakFlag = True
                break
            status = predator.update(agent.position)
            if status != 0:
                breakFlag = True
                break

        if breakFlag:
            if status == 1:
                success = success + 1
            else:
                deaths = deaths + 1
        else:
            timeouts = timeouts + 1
        graph = gr.Graph(config["GRAPH_SIZE"], config)
        graph.create()
        agent_start = random.randrange(0, config["GRAPH_SIZE"])
        agent = A1.Agent1(graph.alist, agent_start, config)
        predator = Pr.Predator(graph.alist, config, agent_start)
        prey = P.Prey(graph.alist, config, agent_start)

    print("Timeouts: " + str(timeouts))
    print("Deaths: " + str(deaths))
    print("Success: " + str(success))


trials()
