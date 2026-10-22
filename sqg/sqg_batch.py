"""Advance the SQG ensemble as one batched array instead of member by member.

The model was already on the GPU, but advance_ens drove it one member at a time:

    xp.stack([advance(E[k], nsteps) for k in range(K)])

so every FFT acted on a single (2, N, N) field.  At N = 128 that is 32768 points -- a 2D FFT
an A100 finishes in about 10 us -- yet the measured cost was 5.9 ms per model step, roughly
600x off and entirely kernel-launch bound.  Advancing all K members as one (K, 2, N, N) array
replaces K tiny transforms by one batched transform.

Only three methods of the vendored model hardcode the leading dimension 2; they are replaced
here by shape-generic versions.  Everything else already broadcasts over a leading ensemble
axis: the spectral operators are (2,1,1), (1,N,M) or (2,N,M), rfft2/irfft2 act on the last two
axes, and advance() itself never inspects the shape.  The vendored file is left untouched --
these are installed onto the class -- so the change is reversible and must be justified by the
equivalence check in test_batch.py, not assumed.
"""

def enable(sqg_module, xp):
    """Install shape-generic invert/specpad/spectrunc onto the model class."""
    cls = sqg_module.SQG
    if getattr(cls, "_batched", False):
        return

    def invert(self, pvspec=None):
        if pvspec is None:
            pvspec = self.pvspec
        # index the LEVEL axis from the right: pvspec is (2,N,M) or (K,2,N,M)
        p0, p1 = pvspec[..., 0, :, :], pvspec[..., 1, :, :]
        psi0 = self.Hovermu*((p1/self.sinhmu) - (p0/self.tanhmu))
        psi1 = self.Hovermu*((p1/self.tanhmu) - (p0/self.sinhmu))
        return xp.stack((psi0, psi1), axis=-3)

    def specpad(self, specarr):
        n = self.N; h = n//2
        out = xp.zeros(specarr.shape[:-2] + (3*n//2, 3*n//4 + 1), specarr.dtype)
        out[..., 0:h, 0:h] = 2.25*specarr[..., 0:h, 0:h]
        out[..., -h:, 0:h] = 2.25*specarr[..., -h:, 0:h]
        out[..., 0:h, h]   = xp.conjugate(2.25*specarr[..., 0:h, -1])
        out[..., -h:, h]   = xp.conjugate(2.25*specarr[..., -h:, -1])
        return out

    def spectrunc(self, specarr):
        n = self.N; h = n//2
        out = xp.zeros(specarr.shape[:-2] + (n, n//2 + 1), specarr.dtype)
        out[..., 0:h, 0:h] = specarr[..., 0:h, 0:h]
        out[..., -h:, 0:h] = specarr[..., -h:, 0:h]
        return out

    cls.invert, cls.specpad, cls.spectrunc = invert, specpad, spectrunc
    cls._batched = True
