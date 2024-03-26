from utils.ImplicitTool import *

# threshlod of cg method
epsilon = 1e-3

# residual vector
r = ti.Vector.field(3, dtype=float, shape=N)
# direction vector
d = ti.Vector.field(3, dtype=float, shape=N)
# for preconditioning of cg method
P = ti.Matrix.field(3, 3, dtype=float, shape=N)
P_inv = ti.Matrix.field(3, 3, dtype=float, shape=N)
# to store A @ d result
q = ti.Vector.field(3, dtype=float, shape=N)
# to store P-1 @ r
s = ti.Vector.field(3, dtype=float, shape=N)


@ti.kernel
def updateP():
    """
    P is the diagonal matrix corresponding to A
    """
    for i in range(N):
        P[i] = A[i, i]
        P_inv[i] = A[i, i].inverse()


@ti.kernel
def calculateDelta0() -> float:
    """
    delta_0
    delta_0 = bT @ P @ b
    """
    delta_0 = 0.0
    for i in range(N):
        delta_0 += b[i] @ P[i] @ b[i]
    return delta_0


@ti.kernel
def initR():
    """
    init r
    r = b - A*dv
    """
    for i in range(N):
        r[i] = b[i]
        for j in range(N):
            r[i] -= A[i, j] @ dv[j]


@ti.kernel
def updateS():
    """
    s = P_inv @ r
    """
    for i in range(N):
        s[i] = ti.Vector([0, 0, 0])

    for i in range(N):
        s[i] = P_inv[i] @ r[i]


@ti.kernel
def initD():
    """
    init conjugate direction
    d = P-1 @ r
    """
    for i in range(N):
        d[i] = s[i]


@ti.kernel
def calculateDelta() -> float:
    """
    delta in every step
    delta = rT @ P-1 @ r
          = rT @ s
    """
    delta = 0.0
    for i in range(N):
        delta += r[i] @ s[i]
    return delta


@ti.kernel
def updateQ():
    """
    q = A @ d
    """
    for i in range(N):
        q[i] = ti.Vector([0, 0, 0])

    for i in range(N):
        for j in range(N):
            q[i] += A[i, j] @ d[j]


@ti.kernel
def calculateAlpha(delta: float) -> float:
    """
    alpha = rT @ P_inv @ r / (dT @ A @ d)
          = rT @ s / (dT @ q)
          = delta / (dT @ q)
    """
    divider = 0.0
    for i in range(N):
        divider += d[i] @ q[i]
    return delta / divider


@ti.kernel
def updateDv(alpha: float):
    """
    dv = dv + alpha * d
    """
    for i in range(N):
        dv[i] = dv[i] + alpha * d[i]


@ti.kernel
def updateR(alpha: float):
    """
    r = r - alpha * A @ d
      = r - alpha @ q
    """
    for i in range(N):
        r[i] = r[i] - alpha * q[i]


@ti.kernel
def updateD(ratio: float):
    """
    d_{i+1} = P_inv @ r_{i+1} + delta / delta_old * d_{i}
    """
    for i in range(N):
        d[i] = s[i] + ratio * d[i]


def CGMethod():
    initDv()
    delta_0 = calculateDelta0()
    initR()
    updateS()
    initD()
    delta = calculateDelta()

    while delta > epsilon * epsilon * delta_0:
        updateQ()
        alpha = calculateAlpha(delta)
        updateDv(alpha)
        updateR(alpha)
        updateS()
        delta_old = delta
        delta = calculateDelta()
        updateD(delta / delta_old)
