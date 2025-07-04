import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from AnalyticalMethods import theis_method, theis_with_no_flow


class ConceptualModel:

    def __init__(self, model_name: str, model_height: float, model_width: float, water_table_depth: float,
                 target_position: float, number_layers):
        self.model_name = model_name
        self.model_height = model_height
        self.model_width = model_width
        self.water_table_depth = water_table_depth
        self.target_position = target_position
        self.number_layers = number_layers

        self.model_x_0 = self.target_position * -1
        self.model_y_0 = self.model_height * -1 + self.water_table_depth

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def add_sky(self, sky_colour="b", sky_thickness: float = 5.0):
        sky = plt.Rectangle((self.model_x_0, self.water_table_depth), width=self.model_width, height=sky_thickness,
                            fc=sky_colour, zorder=0, alpha=0.1)
        self.ax.add_patch(sky)

    def add_layers(self, layers_thickness: list, layers_colours: list, layers_hatching: list, layer_names: list):
        total_thickness = self.water_table_depth
        for i in range(self.number_layers):
            total_thickness -= layers_thickness[i]
            layer = plt.Rectangle(
                (self.model_x_0, total_thickness),  # Anchor point (x, y)
                width=self.model_width,
                height=layers_thickness[i],
                fc=layers_colours[i],  # Colour
                zorder=0,
                alpha=0.9,
                hatch=layers_hatching[i],
                label=layer_names[i]
            )
            self.ax.add_patch(layer)

    def add_pump_borehole(self, abstraction_rate: float, well_x_position: float, borehole_depth: float = None,
                          screen_top: float = None, screen_bottom: float = None):
        # Add the borehole
        self.add_borehole(well_x_position=well_x_position, borehole_depth=borehole_depth)
        # Add the borehole head
        well_head = plt.Rectangle((well_x_position - 2, self.water_table_depth),
                                  width=4, height=2.5, fc=np.array([200, 200, 200]) / 255, zorder=2, ec="k")
        self.ax.add_patch(well_head)
        # Add the screened section
        self.add_piezometer(well_x_position=well_x_position, screen_top=screen_top, screen_bottom=screen_bottom)
        # Add the pumping information
        pumping_arrow = plt.Arrow(x=well_x_position + 2, y=self.water_table_depth + 1.5, dx=5, dy=0, color="#00035b")
        self.ax.add_patch(pumping_arrow)
        self.ax.text(x=well_x_position + 7, y=self.water_table_depth + 1.5,
                     s=f"Q = {abstraction_rate} m\N{SUPERSCRIPT THREE}/d", fontsize="large")

    def add_borehole(self, well_x_position: float, borehole_depth: float = None):
        if borehole_depth is None:
            borehole_depth = self.model_height
        borehole = plt.Rectangle((well_x_position - 1.5, self.water_table_depth - borehole_depth), width=3,
                                 height=borehole_depth, fc=np.array([200, 200, 200]) / 255, zorder=1)
        self.ax.add_patch(borehole)

    def add_piezometer(self, well_x_position: float, screen_top: float = None, screen_bottom: float = None,
                       piezometer_name: str = None):
        if screen_top is None:
            screen_top = self.water_table_depth
        if screen_bottom is None:
            screen_bottom = self.model_height
        screen = plt.Rectangle((well_x_position - 1.5, self.water_table_depth - screen_bottom),
                               width=3,
                               height=screen_bottom - screen_top,
                               fc=np.array([200, 200, 200]) / 255,
                               alpha=1, zorder=2, ec="k", ls="--")
        screen.set_linewidth(2)
        self.ax.add_patch(screen)
        if piezometer_name is not None:
            self.ax.text(well_x_position + 4,
                         self.water_table_depth - (screen_bottom + screen_top) / 2,
                         s=f"{piezometer_name}", bbox={"fc": "w"})

    def add_no_flow(self, x_position: float, thickness: float = 5, hatching: str = "*"):
        no_flow = plt.Rectangle((x_position - thickness / 2, self.model_y_0),
                                width=thickness,
                                height=self.model_height,
                                fc=np.array([209, 109, 127]) / 255,
                                zorder=1, alpha=0.9, hatch=hatching, label="No-flow Boundary")
        self.ax.add_patch(no_flow)

    def plot_information(self):
        surface_line = plt.Line2D(xdata=[self.model_x_0, self.model_width - self.model_x_0],
                                  ydata=[self.water_table_depth, self.water_table_depth], color="k")
        self.ax.add_line(surface_line)

        water_table = plt.Line2D(xdata=[self.model_x_0, self.model_width - self.model_x_0], ydata=[0, 0], color="b",
                                 label="Water table", linestyle="--")
        self.ax.add_line(water_table)
        #self.ax.text(self.model_x_0 + 2, 0.5, s="Water Table", fontsize="large", color="b", bbox={"fc": "w"})

        self.ax.set_xlim([self.model_x_0, self.model_width + self.model_x_0])
        self.ax.set_ylim([self.model_y_0, 7])
        self.ax.set_xlabel("Distance [m]")
        self.ax.set_ylabel("Relative height [m]")
        self.ax.set_title(f"Conceptual Model - {self.model_name}")

        self.ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()

    def save_model(self, save_path: str = None):
        if save_path is None:
            self.fig.savefig(f"{self.model_name}.png", bbox_inches="tight")
        else:
            self.fig.savefig(f"{save_path}{self.model_name}.png", bbox_inches="tight")


class AnimatedConceptualModel:
    def __init__(self, fig: plt.Figure, ax: plt.Axes, time: list, space: list, zero_position: float,
                 abstraction_rate: float, transmissivity: float, storativity: float,
                 no_flow: bool = False, no_flow_distance: float = 0.0):
        # Get the figure
        self.fig = fig
        self.ax = ax
        # Get additional information on the conceptual model
        self.zero_position = zero_position
        self.abstraction_rate = abstraction_rate
        self.transmissivity = transmissivity
        self.storativity = storativity
        self.no_flow = no_flow
        self.no_flow_distance = no_flow_distance
        # Add an empty line object
        plotted_lines = ax.plot([], color='blue', linestyle='-', linewidth=2)
        self.plotted_line = plotted_lines[0]
        # Get the temporal and spatial information
        self.space = space
        self.time = time
        self.pump_off_time = max(time)
        self.pump_off_index = len(time)
        # Create the recovery time steps
        for index in range(len(time)):
            self.time.append(self.pump_off_time + time[index])

    def create_animation(self, output_file: str):
        pump_off_drawdown = []

        def AnimationFunction(frame):
            if frame > self.pump_off_index:
                func_pump_off = True
                x = self.time[frame] - self.pump_off_time
                abstraction_rate = -1 * self.abstraction_rate
            else:
                func_pump_off = False
                x = self.time[frame]
                abstraction_rate = self.abstraction_rate
            drawdown = []
            # Get drawdown values along the section
            for space_index in range(len(self.space)):
                if self.no_flow:
                    drawdown_calc = theis_with_no_flow(time=x,
                                                       transmissivity=self.transmissivity,
                                                       storativity=self.storativity,
                                                       radius=self.space[space_index],
                                                       abstraction_rate=abstraction_rate,
                                                       pump_off=func_pump_off,
                                                       no_flow_distance=self.no_flow_distance)
                else:
                    drawdown_calc = theis_method(time=x,
                                                 transmissivity=self.transmissivity,
                                                 storativity=self.storativity,
                                                 radius=abs(self.space[space_index]),
                                                 abstraction_rate=abstraction_rate)

                if frame == self.pump_off_index:
                    pump_off_drawdown.append(0 - drawdown_calc)
                if frame > self.pump_off_index:
                    drawdown.append(pump_off_drawdown[space_index] - drawdown_calc)
                else:
                    drawdown.append(0 - drawdown_calc)
            # Plot as line that changes on frame
            self.plotted_line.set_data((self.space, drawdown))

        # Create the animation
        anim_created = FuncAnimation(self.fig, AnimationFunction, repeat=True, frames=len(self.time), interval=20)
        # Convert the animation using Pillow to a gif
        writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # Save the gif file to disk
        anim_created.save(f"{output_file}.gif", writer=writer)

    def time_series(self, radius: float) -> plt.Figure:
        calc_drawdown = []
        calc_pump_off_drawdown = 0.0
        pump_off = False
        for index in range(len(self.time)):
            dd = theis_method(time=self.time[index], transmissivity=self.transmissivity, storativity=self.storativity,
                              radius=radius, abstraction_rate=self.abstraction_rate, pump_off=pump_off)
            if index == self.pump_off_index:
                pump_off = True
                calc_pump_off_drawdown = dd
            if index > self.pump_off_index:
                calc_drawdown.append(calc_pump_off_drawdown + dd)
            else:
                calc_drawdown.append(dd)

        fig, ax = plt.subplots()
        ax.plot(self.time, calc_drawdown)
        return fig
