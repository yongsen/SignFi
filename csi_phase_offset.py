from scipy.optimize import curve_fit

# x (tx and subcarrier index) for curve fitting
idx_tx_subc = np.zeros(shape=(2, 3, 3, 30))
for tx in range(3):
    for rx in range(3):
        for k in range(30):
            idx_tx_subc[0, tx, rx, k] = (tx+2)/3 - 2 # tx index, reordered
            idx_tx_subc[1, tx, rx, k] = -58 + 4*k # subcarrier index
idx_tx_subc = np.reshape(idx_tx_subc, (2, 270))

def func(x, a, b, c):
    """Phase offsets function
    x[0]: transmit antenna index
    x[1]: subcarrier index
    """
    return a*x[0] *x[1] + b*x[1] + c

def remove_phase_offset(csi):
    csi = np.array(csi)
    csi_shape = csi.shape
    csi_abs = np.abs(csi)
    csi_ang = np.angle(csi)
    phase = csi_ang[0, ...] # get phase offsets from the 1st packet
    for tx in range(csi_shape[1]): # tx
        for rx in range(csi_shape[2]): # rx
            phase[tx, rx, :] = unwrap(phase[tx, rx, :])

    phase = phase. flatten()
    popt, pcov = curve_fit(func, idx_tx_subc, phase)

    phase = func(idx_tx_subc, *popt)
    phase = np.reshape(phase, (csi_shape(1),csi_shape [ 2 ], csi_shape [3]))

    pdp = np.zeros(shape=csi_shape, dtype=np.complex)
    for t in range (csi_shape[0]): # time
        for tx in range(csi_shape[1]): # tx
            for rx in range(csi_shape[2]): # rx
                csi_ang[t, tx, rx, :] = unwrap(csi_ang[t, tx, rx, :])
                csi_ang[t, tx, rx, :] -= phase[tx, rx, :]

                csi_new = csi_abs[t, tx, rx, : ]*np.exp(1j*csi_ang[t, tx, rx, :])
                pdp[t, tx,rx, :] = np.fft.ifft(csi_new)
    return csi_abs, csi_ang, pdp
