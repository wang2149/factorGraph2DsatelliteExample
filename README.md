# factorGraph2DsatelliteExample
Comparison of the performance of EKF and FG after incorporating linear velocity observation.
This repository is based on the 2D satellite orbit positioning example proposed in Factor Graphs for Navigation Applications: A Tutorial. In the original framework that only contains azimuth observations, the linear velocity observation of the satellite in orbit is added. In the two state estimation methods, extended Kalman filter (EKF) and factor graph (FG), the measurement equations, Jacobian matrices and update formulas were derived and the codes were adjusted accordingly. Through simulation verification, the performance indicators of the two schemes, azimuth only observation and azimuth + velocity observation, were systematically compared to explore under what conditions the addition of linear velocity observations can significantly improve the effect of spacecraft orbit estimation.

The folder with velocity suffix has the linear velocity observation added, and the folder without velocity suffix is ​​the original one.
Original code version is from:https://github.com/cntaylor/factorGraph2DsatelliteExample
