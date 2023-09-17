import numpy as np
# Calculate Longitudinal force in pure slip conditions knowing the 
# 14 parametrs of the magic tyre model

b0 = 2.139
b1 = 0.0045
b2 = -0.934
b3 = 1.971
b4 = 6.081
b5 = 0.0654
b6 = -0.0014
b7 = 0.04
b8 = 2.229
b9 = 9.716
b10 = 5.626
b11 = 3.2e-6
b12 = -8.7e-5
b13 = 0.649
Fz = 26.663
slip_angles = np.arange(-0.75, 0.76, 0.02)

Y_pure = []
for slip_angle in slip_angles:

    D = b1 * pow(Fz, 2) + b2 * Fz
    C = b0
    B = ((b3 * pow(Fz, 2) + b4 * Fz) * np.exp(-b5*Fz))/(C*D)
    Shx = b9 * Fz + b10
    Svx = b11 * Fz + b12
    E = (b6 * pow(Fz, 2) + b7*Fz + b8) * (1 - b13 * np.sign(slip_angle + Shx))

    x = slip_angle
    y = D * np.sin(C * np.arctan(B * x - E *
                    (B * x - np.arctan(B*x)))) + Svx
    y = y/(x+Shx)
    Y_pure.append(y)    # Longitudinal force in pure slip condition
    
np.savetxt('Y_mes.txt', Y_pure, fmt='%.8f')
np.savetxt('Fz.txt', [Fz], fmt="%.3f")
np.savetxt('slipAngle.txt', slip_angles, fmt='%.2f')

