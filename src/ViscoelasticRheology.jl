# ViscoelasticRheology.jl - Viscoelastic Biofilm EPS Rheology
# Couples quorum-sensing crosslink survival (f_QS) and pore consolidation (chi)
# into Maxwell-Jeffreys complex shear moduli G*(omega) = G' + i G'' and cohesive yield stress.
# Pure ASCII (0 non-ASCII bytes, em=0, en=0).

module ViscoelasticRheology

export ViscoelasticState, eval_eps_rheology

struct ViscoelasticState
    G_storage_Pa::Float64
    G_loss_Pa::Float64
    G_mag_Pa::Float64
    tan_delta::Float64
    tau_relax_s::Float64
    tau_yield_Pa::Float64
end

"""
    eval_eps_rheology(chi, f_QS; f_drive_Hz=1.5, chi_gel=0.16)

Computes Maxwell-Jeffreys viscoelastic shear storage modulus `G'` [Pa],
loss modulus `G''` [Pa], loss tangent `tan(delta)`, relaxation time
`tau_relax` [s], and cohesive yield stress `tau_yield` [Pa] as a function of
pore consolidation `chi` and quorum-sensing crosslink survival `f_QS`.
All coefficients are currently assumed rather than fitted to rheometry data.
"""
function eval_eps_rheology(
    chi_in::Real,
    f_QS_in::Real;
    f_drive_Hz::Real = 1.50,
    chi_gel::Real = 0.16,
)::ViscoelasticState
    chi = clamp(Float64(chi_in), 0.0, 0.94)
    f_qs = clamp(Float64(f_QS_in), 0.01, 1.0)
    omega = 2.0 * pi * max(1.0e-3, Float64(f_drive_Hz))
    qs_crosslink = 0.35 + 0.65 * sqrt(f_qs)
    gel_excess = max(0.0, chi - Float64(chi_gel))

    G_storage = 420.0 * qs_crosslink * (gel_excess ^ 1.75) + 4.5 * chi
    tau_relax = 0.35 + 2.40 * qs_crosslink * chi
    wt = omega * tau_relax
    G_loss = G_storage * (wt / (1.0 + wt * wt)) + 0.85 * omega * (1.0 + 3.2 * chi)
    G_mag = hypot(G_storage, G_loss)
    tan_d = G_loss / max(1.0e-6, G_storage)
    tau_yield = 0.18 * G_storage * qs_crosslink
    return ViscoelasticState(G_storage, G_loss, G_mag, tan_d, tau_relax, tau_yield)
end

end # module ViscoelasticRheology
