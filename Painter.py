import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def init_frame():
	root = tk.Tk()
	root.title("The Circle of Life")
	root.resizable(False, False)
	return root

def init_canvas(frame, f):
	canvas = FigureCanvasTkAgg(f, master=frame)
	canvas.draw()
	canvas.get_tk_widget().pack(side = tk.TOP, fill=tk.BOTH, expand = 1)
	return canvas

def set_up_figure(networkx_graph):
	fig = plt.figure(figsize=(13,7))
	axis = fig.add_subplot(111)
	plt.axis('off')
	node_positions = nx.kamada_kawai_layout(networkx_graph)
	nx.draw_networkx(networkx_graph,pos=node_positions,ax=axis)
	xlim = axis.get_xlim()
	ylim = axis.get_ylim()
	fig.tight_layout()
	return fig, axis, node_positions, xlim, ylim


def draw_next_graph(frame, canvas, networkx_graph, fig, axis, node_positions, xlim, ylim, predator, prey, agent):
    axis.cla()
    color_map = ['blue' if node == agent.position else 'red' if node == predator.position else 'green' if node == prey.position else 'yellow' for node in networkx_graph] 
    nx.draw_networkx(networkx_graph, node_positions, ax=axis, node_color=color_map)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    plt.axis('off')
    canvas.draw()
