import math
from scipy.special import exp1  # For the exponential integral E1(x)


# Calculate u
def dimensionless_time_parameter(time: float, radius: float, storativity: float, transmissivity: float) -> float:
    """
    Calculates the dimensionless time parameter for use in the Theis equation.

    Args:
        time (float): The time in days since pumping started.
        transmissivity (float): The transmissivity of the aquifer in m2/d.
        storativity (float): The storativity of the aquifer.
        radius (float): The distance between the pumping well and the point of observation.

    Returns:
        float: The dimensionless time parameter for the well at time x.

    """
    return (radius**2 * storativity) / (4 * transmissivity * time)


# Theis method
def theis_method(time: float, transmissivity: float, storativity: float, radius: float, abstraction_rate: float) -> float:
    """
    Calculates the drawdown in a well using the Theis equation.

    Args:
        x (float): The time in minutes since pumping started.
        transmissivity (float): The transmissivity of the aquifer in m2/d.
        storativity (float): The storativity of the aquifer.
        radius (float): The distance between the pump and the point of observation.
        abstraction_rate (float): The rate at which water is being pumped from the aquifer.

    Returns:
        float: The calculated drawdown in the well at time x.

    """
    u = dimensionless_time_parameter(radius=radius, storativity=storativity, transmissivity=transmissivity, time=(time / 1440))
    w_u = exp1(u)  # Calculate the exponential integral for the dimensionless time parameter E1(u)
    drawdown = (abstraction_rate / (4 * math.pi * transmissivity)) * w_u
    return drawdown


# Theis with the addition of a no-flow boundary
def theis_with_no_flow(time: float, transmissivity: float, storativity: float, radius: float, abstraction_rate: float,
                       pump_off: bool, no_flow_distance: float = 0.0, ) -> float:
    """
    Calculates the drawdown in a well using the Theis equation in the presence of a single no-flow boundary at a specified distance.

    Args:
        time (float): The time in minutes since pumping started.
        transmissivity (float): The transmissivity of the aquifer in m2/d.
        storativity (float): The storativity of the aquifer.
        radius (float): The distance between the pumping well and the point of observation.
        abstraction_rate (float): The rate at which water is being pumped from the aquifer.
        pump_off (bool): Is the pump turned off for this time step.
        no_flow_distance (float): The distance in meters to the no-flow boundary from the pumping well.

    Returns:
        float: The calculated drawdown in the well at time x.

    """
    distance_to_no_flow = no_flow_distance + (no_flow_distance - radius)
    drawdown_hole = theis_method(time=time, transmissivity=transmissivity, storativity=storativity, radius=abs(radius),
                                 abstraction_rate=abstraction_rate)
    drawdown_nflw = theis_method(time=time, transmissivity=transmissivity, storativity=storativity,
                                 radius=distance_to_no_flow, abstraction_rate=abstraction_rate)
    if pump_off:
        drawdown_calc = drawdown_hole - drawdown_nflw
    else:
        drawdown_calc = drawdown_hole + drawdown_nflw
    return drawdown_calc
