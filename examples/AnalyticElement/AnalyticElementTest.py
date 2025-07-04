import numpy as np
import matplotlib.pyplot as plt

from cAnalyticElement import (AnalyticElementModel, AnalyticWell, AnalyticUniformFlow, AnalyticLineSink,
                                    AnalyticHeadWell, AnalyticConstant, AnalyticHeadLineSink, AnalyticAreaSink)

# Test analytic element method
# Wells test
test_model = AnalyticElementModel(transmissivity=200)
well_one = AnalyticWell(test_model, flow_rate=200.0, well_radius=0.08, xw=50.0, yw=0.0)  # Abstraction
well_two = AnalyticWell(test_model, flow_rate=-200.0, well_radius=0.08, xw=-50.0, yw=0.0)  # Injection
flow_field = AnalyticUniformFlow(test_model, gradient=0.002, angle=-45.0)

xg, yg = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100))
pot = test_model.potential(xg, yg)
psi = test_model.stream_function(xg, yg)

fig_grad = plt.figure()
plt.imshow(psi, interpolation='none')
plt.savefig("./output/AnalyticElementGradientPsi.png")
plt.close(fig_grad)

fig_plot = plt.figure()
plt.contour(xg, yg, pot, 20, colors='blue')
plt.contour(xg, yg, psi, 20, colors='orange')
fig_plot.savefig("./output/AnalyticElementPlot.png")
plt.close(fig_plot)

# Constant head test
constant_head_model = AnalyticElementModel(transmissivity=100)
test_well = AnalyticWell(constant_head_model, xw=0, yw=0, flow_rate=100, well_radius=0.3)
head_well = AnalyticHeadWell(constant_head_model, xw=400.0, yw=300.0, rw=0.3, hw=20.0)
uniform_flow = AnalyticUniformFlow(constant_head_model, gradient=0.002, angle=0.0)
rf = AnalyticConstant(constant_head_model, xc=400.0, yc=0.0, hc=22.0)
constant_head_model.solve_model()

print(f"Computed head at discharge well: {constant_head_model.head(0.0, 0.0):.2f} m")
print(f"Computed head at constant-head well: {constant_head_model.head(400.3, 300.0):.2f} m")
print(f"Computed discharge at constant-head well: {head_well.parameter:.2f} m3/d")
print(f"Computed head at reference point: {constant_head_model.head(400.0, 0.0):.2f} m")

xg, yg = np.meshgrid(np.linspace(-200, 600, 100), np.linspace(-100, 500, 100))
head = constant_head_model.head(xg, yg)
psi = constant_head_model.stream_function(xg, yg)

print(f"Min and max head on grid: {head.min():.2f}, {head.max():.2f}")
print(f"Min and max psi on grid: {psi.min():.2f}, {psi.max():.2f}")

fig_ch_plot = plt.figure()
plt.contour(xg, yg, head, np.arange(20, 24, 0.2), colors='blue')
plt.contour(xg, yg, psi, np.arange(-200, 100, head_well.parameter / 20), colors='orange')
fig_ch_plot.savefig("./output/AnalyticConstantHead.png")
plt.close(fig_ch_plot)


# Line sink test
line_test_model = AnalyticElementModel(transmissivity=100)
line_sink_feature = AnalyticLineSink(line_test_model, x0=-200, y0=-150, x1=200, y1=150, sigma=0.1)

xg, yg = np.meshgrid(np.linspace(-400, 400, 100), np.linspace(-300, 300, 100))
h = line_test_model.head(xg, yg)
psi = line_test_model.stream_function(xg, yg)
xs, ys = np.linspace(-200, 200, 100), np.linspace(-150, 150, 100)
hs = line_test_model.head(xs, ys)

plt.subplot(121, aspect=1)
plt.contour(xg, yg, h, 10, colors='blue', label='Head')
plt.contour(xg, yg, psi, 20, colors='orange', label='Streamlines')
plt.plot([-200, 200], [-150, 150], 'black', label="Line sink")
plt.subplot(122)
plt.plot(np.sqrt((xs + 200)**2 + (ys + 150)**2), hs)
plt.savefig("./output/AnalyticElementLineSink.png")
plt.clf()

# Head Link Sink Test
hriver = 10
head_line_model = AnalyticElementModel(transmissivity=100)
rf = AnalyticConstant(head_line_model, xc=0.0, yc=1000.0, hc=hriver + 2.0)
w = AnalyticHeadWell(head_line_model, xw=0.0, yw=100.0, rw=0.3, hw=hriver - 2.0)
xls = np.linspace(-1600.00, 1600.00, 101)
xls = np.hstack((np.arange(-1600.0, -400.0, 200), np.arange(-400.0, 400.0, 50.0), np.arange(400.0, 1601.0, 200.0)))
yls = 50 * np.sin(np.pi * xls / 400)

for i in range(len(xls) - 1):
    AnalyticHeadLineSink(head_line_model, x0=xls[i], y0=yls[i], x1=xls[i + 1], y1=yls[i + 1], hc=hriver)

head_line_model.solve_model()

xg1, yg1 = np.meshgrid(np.linspace(-1800, 1800, 100), np.linspace(-1200, 1200, 100))
h1 = head_line_model.head(xg1, yg1)
xg2, yg2 = np.meshgrid(np.linspace(-400, 400, 100), np.linspace(-100, 400, 100))
h2 = head_line_model.head(xg2, yg2)

plt.subplot(121, aspect=1)
plt.contour(xg1, yg1, h1, 10, colors='blue')
plt.plot(xls, yls, 'black')
plt.subplot(122, aspect=1, xlim=(-400, 400))
plt.contour(xg2, yg2, h2, 10, colors='blue')
plt.plot(xls, yls, 'black')
plt.savefig("./output/AnalyticElementHeadLineSink.png")
plt.clf()

# Test area sink
area_sink_model = AnalyticElementModel(transmissivity=100)
rf = AnalyticConstant(area_sink_model, xc=0.0, yc=0.0, hc=20.0)
area_sink_1 = AnalyticAreaSink(area_sink_model, xc=-500.0, yc=0.0, N=0.001, R=500.0)
area_sink_2 = AnalyticAreaSink(area_sink_model, xc=500.0, yc=0.0, N=-0.001, R=500.0)

area_sink_model.solve_model()

xg, yg = np.meshgrid(np.linspace(-1500, 1500, 100), np.linspace(-800, 800, 101))
h = area_sink_model.head(xg, yg)

plt.subplot(121, aspect=1)
plt.contour(xg, yg, h, 20, colors='blue')
plt.subplot(122)
plt.plot(xg[0], h[50], 'black')
plt.savefig("./output/AnalyticElementAreaSink.png")
plt.clf()

# Real test
xls0 = [0, 100, 200, 400, 600, 800, 1000, 1100, 1200]
yls0 = [200, 200, 100, 100, 0, 0, 100, 300, 450]
hls0 = np.linspace(39, 40.4, 8)
xls1 = [0, 0, 200, 400, 600, 800, 1000, 1100, 1200]
yls1 = [200, 400, 600, 600, 700, 700, 750, 800, 850]
hls1 = np.linspace(39, 40.4, 8)

real_model = AnalyticElementModel(transmissivity=100)
rf = AnalyticConstant(real_model, xc=0, yc=800, hc=39.5)
well_one = AnalyticWell(real_model, xw=500, yw=250, flow_rate=100.0, well_radius=0.06)
well_two = AnalyticWell(real_model, xw=800, yw=500, flow_rate=100.0, well_radius=0.06)

for i in range(len(hls0)):
    AnalyticHeadLineSink(real_model, xls0[i], yls0[i], xls0[i + 1], yls0[i + 1], hls0[i])
for i in range(len(hls1)):
    AnalyticHeadLineSink(real_model, xls1[i], yls1[i], xls1[i + 1], yls1[i + 1], hls1[i])

ar = AnalyticAreaSink(real_model, xc=600, yc=400, N=0.001, R=700)

real_model.solve_model()

xg, yg = np.meshgrid(np.linspace(-100, 1300, 100), np.linspace(-100, 900, 100))
h = real_model.head(xg, yg)

plt.subplot(111, aspect=1)
cs = plt.contour(xg, yg, h, np.arange(38, 41, 0.2), colors='blue')
plt.clabel(cs, fmt='1%.1f', fontsize='smaller')
plt.contour(xg, yg, h, np.arange(38.1, 41, 0.2), colors='blue')
plt.plot(xls0, yls0, 'green', label="River One")
plt.plot(xls1, yls1, 'purple', label="River Two")
plt.plot([well_one.zetaw.real, well_two.zetaw.real], [well_one.zetaw.imag, well_two.zetaw.imag], 'k.', label="Wells")
plt.legend()
plt.savefig("./output/AnalyticElementRealTest.png")
plt.clf()

########################################################################################################################
# Test mirrors
T = 100
hc = 0
d = 100
Q = [100, 100, -100, -100]
xw = [50, -50, 50, -50]
yw = [50, 50, -50, -50]
phic = T * hc

xg, yg = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100))
pot = phic
psi = 0.0
Qx = 0.0
Qy = 0.0

for n in range(len(Q)):
    rsq = (xg - xw[n]) ** 2 + (yg - yw[n]) ** 2
    pot += Q[n] / (4 * np.pi) * np.log(rsq)
    psi += Q[n] / (2 * np.pi) * np.arctan2(yg - yw[n], xg - xw[n])
    Qx += -Q[n] / (2 * np.pi) * (xg - xw[n]) / rsq
    Qy += -Q[n] / (2 * np.pi) * (yg - yw[n]) / rsq

plt.figure()
plt.subplot(121, aspect=1)
plt.contour(xg, yg, pot, 10, colors='blue')
plt.contour(xg, yg, psi, 10, colors='orange')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.subplot(122, aspect=1)
plt.streamplot(xg, yg, Qx, Qy)
plt.savefig("./output/MirrorTestDouble.png")
