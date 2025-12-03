import numpy as np

def RFsample(L = 1.0, N = 128, kappa = 3.0, alpha = 2.0):
    rng = np.random.default_rng()

    dx = dy = L / N
    dk = 2 * np.pi / L  # Frequency spacing in both x and y

    # 2D wavenumbers
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    # Spectral density
    S = 1.0 / (kappa**2 + K2)**alpha

    # Generate complex Gaussian white noise with Hermitian symmetry
    z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))

    # Impose Hermitian symmetry so the field is real
    z = 0.5 * (z + np.conj(np.flip(np.flip(z, axis=0), axis=1)))

    # Spectrum times sqrt(dk^2)
    dk2 = dk * dk
    u_hat = np.sqrt(S * dk2) * z

    # Inverse FFT and rescale by N² to match amplitude
    u = np.fft.ifft2(u_hat).real * (N**2)
    u -= np.mean(u) # Ensure mean zero

    return u

if __name__ == "__main__":
    sample = RFsample()
    import matplotlib.pyplot as plt
    plt.subplots(figsize= (6,6))
    plt.imshow(sample, extent=[0,1,0,1])
    plt.colorbar()
    plt.tight_layout()
    plt.show()
