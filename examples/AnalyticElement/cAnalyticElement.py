import numpy as np

# Credit: "Analytical Groundwater Modeling: Theory and Applications Using Python" 
# by Mark Bakker and Vincent Post, published by CRC Press.

class AnalyticElementModel:

    def __init__(self, transmissivity: float):
        self.elements = []
        self.transmissivity = transmissivity

    def omega(self, x: float, y: float) -> complex:
        omega = 0 + 0j
        for element in self.elements:
            omega += element.omega(x, y)
        return omega

    def potential(self, x, y) -> float:
        return self.omega(x, y).real

    def stream_function(self, x: float, y: float) -> complex:
        return self.omega(x, y).imag

    def head(self, x: float, y: float) -> float:
        return self.potential(x, y) / self.transmissivity

    def solve_model(self):
        element_solve = [element for element in self.elements if element.number_unknowns == 1]
        number_unknowns = len(element_solve)
        matrix = np.zeros((number_unknowns, number_unknowns))
        rhs = np.zeros(number_unknowns)
        for i in range(number_unknowns):
            matrix[i], rhs[i] = element_solve[i].equation()
        solution = np.linalg.solve(matrix, rhs)
        for i in range(number_unknowns):
            element_solve[i].parameter = solution[i]


class AnalyticElement:
    def __init__(self, input_model: AnalyticElementModel, param: float):
        self.model = input_model
        self.parameter = param
        self.number_unknowns = 0
        self.model.elements.append(self)

    def omega(self, x: float, y: float) -> complex:
        return self.parameter * self.omegainf(x, y)

    def potential(self, x: float, y: float) -> float:
        return self.omega(x, y).real

    def potinf(self, x: float, y: float) -> float:
        return self.omegainf(x, y).real


class AnalyticWell(AnalyticElement):
    def __init__(self, input_model: AnalyticElementModel, flow_rate: float, well_radius: float, xw:float, yw:float):
        AnalyticElement.__init__(self, input_model, flow_rate)
        self.zetaw = xw + 1j * yw
        self.well_radius = well_radius

    def omegainf(self, x: float, y: float) -> complex:
        zminzw = x + 1j * y - self.zetaw
        zminzw = np.where(np.abs(zminzw) < self.well_radius, self.well_radius, zminzw)
        return 1 / (2 * np.pi) * np.log(zminzw)


class AnalyticUniformFlow(AnalyticElement):
    def __init__(self, input_model: AnalyticElementModel, gradient: float, angle: float):
        AnalyticElement.__init__(self, input_model, input_model.transmissivity * gradient)
        self.udir = np.exp(-1j * np.deg2rad(angle))

    def omegainf(self, x: float, y: float) -> complex:
        return -self.udir * (x + y * 1j)


class AnalyticHeadEquation:
    def equation(self):
        row = []
        rhs = self.pc
        for element in self.model.elements:
            if element.number_unknowns == 1:
                row.append(element.potinf(self.xc, self.yc))
            else:
                rhs -= element.potential(self.xc, self.yc)
        return row, rhs


class AnalyticHeadWell(AnalyticWell, AnalyticHeadEquation):
    def __init__(self, input_model: AnalyticElementModel, xw: float, yw: float, rw: float, hw: float):
        AnalyticWell.__init__(self, input_model=input_model, xw=xw, yw=yw, flow_rate=0, well_radius=rw)
        self.xc, self.yc = xw + rw, yw
        self.pc = self.model.transmissivity * hw
        self.number_unknowns = 1


class AnalyticConstant(AnalyticElement, AnalyticHeadEquation):
    def __init__(self, input_model: AnalyticElementModel, xc: float, yc: float, hc: float):
        AnalyticElement.__init__(self, input_model=input_model, param=0.0)
        self.xc, self.yc = xc, yc
        self.pc = self.model.transmissivity * hc
        self.number_unknowns = 1

    def omegainf(self, x: float, y: float) -> complex:
        return np.ones_like(x, dtype='complex')


class AnalyticLineSink(AnalyticElement):
    def __init__(self, input_model: AnalyticElementModel, x0: float = 0.0, y0: float = 0.0, x1: float = 1.0, y1: float = 1.0, sigma: float = 1.0):
        AnalyticElement.__init__(self, input_model, sigma)
        self.z0 = x0 + y0 * 1j
        self.z1 = x1 + y1 * 1j
        self.L = np.abs(self.z1 - self.z0)

    def omegainf(self, x: float, y: float) -> complex:
        zeta = x + y * 1j
        Z = (2 * zeta - (self.z0 + self.z1)) / (self.z1 - self.z0)
        Zp1 = np.where(np.abs(Z + 1) < 1e-12, 1e-12, Z + 1)
        Zm1 = np.where(np.abs(Z - 1) < 1e-12, 1e-12, Z - 1)
        return self.L / (4 * np.pi) * (Zp1 * np.log(Zp1) - Zm1 * np.log(Zm1))


class AnalyticHeadLineSink(AnalyticLineSink, AnalyticHeadEquation):
    def __init__(self, input_model: AnalyticElementModel, x0: float = 0.0, y0: float = 0.0, x1: float = 1.0, y1: float = 1.0, hc: float = 1.0):
        AnalyticLineSink.__init__(self, input_model, x0, y0, x1, y1, hc)
        self.xc = 0.5 * (x0 + x1)
        self.yc = 0.5 * (y0 + y1)
        self.pc = self.model.transmissivity * hc
        self.number_unknowns = 1


class AnalyticAreaSink(AnalyticElement):
    def __init__(self, input_model: AnalyticElementModel, xc: float = 0.0, yc: float = 0.0, N: float = 0.001, R: float = 100.0):
        AnalyticElement.__init__(self, input_model, N)
        self.xc, self.yc = xc, yc
        self.R = R

    def omegainf(self, x, y):
        r = np.atleast_1d(np.sqrt((x - self.xc) ** 2 + (y - self.yc) ** 2))
        phi = np.zeros(r.shape)
        phi[r < self.R] = -0.25 * (r[r < self.R] ** 2 - self.R ** 2)
        phi[r >= self.R] = -self.R ** 2 / 2 * np.log(r[r >= self.R] / self.R)
        return phi + 0j

