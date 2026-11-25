module CoreTransport

export MODE_BIOCHAR_MFC, MODE_BIO_H2_MEC
export CoreParams, PyrolysisBiocharParams, EdgeSensitivities
export char_area_resistance, default_params, biochar_consolidated_params, edge_weight_sensitivities
export step_core, simulate_trajectory, selftest

const MODE_BIOCHAR_MFC = Int32(1)
const MODE_BIO_H2_MEC  = Int32(2)

const F_CONST = 96485.332f0
const R_GAS   = 8.3144626f0
const T_STD   = 298.15f0
const FRT     = F_CONST / (R_GAS * T_STD)  # ~38.9217 V^-1

struct CoreParams
    mode::Int32
    E0_V::Float32
    j_scale::Float32     # j0 (BV) or j_max (Nernst-Monod) [mA/cm^2]
    alpha::Float32       # symmetry factor (BV) or Nernst slope multiplier
    K_half::Float32      # Monod / mass-transfer half-saturation [mol/L or norm]
    R_ohm::Float32       # area-specific ohmic resistance [Ohm*cm^2]
    k_para::Float32      # parasitic rate constant [mA/cm^2]
    n_e::Float32         # electrons transferred per mole of reactant
    tau_rc::Float32      # double-layer relaxation time [s]
    v_ratio::Float32     # A_elec / V_bulk conversion scale [1/(mA*s)]
    theta_cap::Float32   # surface/biofilm capacity scale [mC/cm^2]
end

struct PyrolysisBiocharParams
    T_pyro_C::Float32    # pyrolysis temperature [400..850 C]
    eps_0::Float32       # initial open char macroporosity [0.35..0.85]
    L_f_um::Float32      # equivalent biofilm thickness [2..120 um]
    chi::Float32         # Bruggeman pore-consolidation modulus [0..0.94]
    eps_eff::Float32     # remaining open porosity eps_0 * (1 - chi)
    R_char::Float32      # solid-matrix char resistance [Ohm*cm^2]
    R_ohm_eff::Float32   # consolidated total resistance [Ohm*cm^2]
    K_half_eff::Float32  # hindered-diffusion half-saturation
    j_max_eff::Float32   # percolated catalytic current scale [mA/cm^2]
    k_para_eff::Float32  # parasitic + basal necrosis loss scale [mA/cm^2]
end

struct EdgeSensitivities
    dRohm_dTpyro::Float32  # Ohm*cm^2 per C (graphitization slope)
    dKhalf_dchi::Float32   # hindered-diffusion stiffness with consolidation chi
    dP_dLf::Float32        # mW/(cm^2 * um) near operating thickness (0 at L_opt)
    detaF_dchi::Float32    # coulombic efficiency sensitivity to throat necking
end

@inline function char_area_resistance(T_pyro_C::Float32)::Float32
    return 9.95f0 + 32.35f0 / (1.0f0 + exp((T_pyro_C - 548.0f0) / 56.0f0))
end

@inline function default_params(mode::Int32)::CoreParams
    if mode == MODE_BIOCHAR_MFC
        return CoreParams(
            MODE_BIOCHAR_MFC,
            -0.30f0, 0.85f0, 1.00f0, 0.15f0, 18.0f0, 0.08f0,
            8.0f0, 3.5f0, 0.00085f0, 420.0f0
        )
    else
        return CoreParams(
            MODE_BIO_H2_MEC,
            -0.41f0, 1.40f0, 0.85f0, 0.10f0, 9.50f0, 0.07f0,
            2.0f0, 2.4f0, 0.00105f0, 340.0f0
        )
    end
end

"""
biochar_consolidated_params(T_pyro_C, eps_0, L_f_um; mode=MODE_BIOCHAR_MFC, ...)

Couples lignocellulosic biochar pyrolysis temperature `T_pyro_C` (C), initial
vascular macroporosity `eps_0`, and biofilm thickness `L_f_um` (um) via the
Bruggeman pore-consolidation modulus `chi in [0, 0.94]`, with optional empirical
baseline overrides `(base_j_scale, base_K_half, base_k_para)` from a calibration receipt.
"""
@inline function biochar_consolidated_params(
    T_pyro_C::Float32,
    eps_0::Float32,
    L_f_um::Float32;
    mode::Int32 = MODE_BIOCHAR_MFC,
    tau_brug::Float32 = 1.80f0,
    L_elec_um::Float32 = 28.0f0,
    R_neck::Float32 = 6.50f0,
    base_j_scale::Float32 = -1.0f0,
    base_K_half::Float32 = -1.0f0,
    base_k_para::Float32 = -1.0f0
)
    base = default_params(mode)
    j0_ref = base_j_scale > 0.0f0 ? base_j_scale : base.j_scale
    K0_ref = base_K_half > 0.0f0 ? base_K_half : base.K_half
    kp_ref = base_k_para > 0.0f0 ? base_k_para : base.k_para

    # 1. Pyrolysis temperature tradeoff (biochar conductivity vs surface quinones)
    f_cond = 1.0f0 / (1.0f0 + exp(-(T_pyro_C - 560.0f0) / 65.0f0))
    z_quin = (T_pyro_C - 500.0f0) / 115.0f0
    f_quin = 1.0f0 + 0.35f0 * exp(-(z_quin * z_quin))
    R_char = char_area_resistance(T_pyro_C)

    # 2. Two-regime consolidation modulus chi in [0, 0.94]
    L_crit_um = 22.0f0 + 68.0f0 * eps_0
    fill_ratio = L_f_um / L_crit_um
    chi_raw = (1.0f0 - exp(-0.82f0 * fill_ratio)) * (0.48f0 + 0.46f0 * tanh(0.65f0 * fill_ratio))
    chi = clamp(chi_raw, 0.0f0, 0.94f0)
    one_minus_chi = max(0.06f0, 1.0f0 - chi)
    eps_eff = eps_0 * one_minus_chi

    # 3. Rivaling curves (percolation vs Bruggeman diffusion & throat necking)
    f_cata = tanh(L_f_um / L_elec_um)
    brug_factor = one_minus_chi ^ tau_brug
    K_half_eff = K0_ref / max(0.04f0, brug_factor)
    R_ohm_eff = R_char + R_neck * (chi * chi) / one_minus_chi + 0.14f0 * max(0.0f0, L_f_um - L_crit_um)
    k_para_eff = kp_ref * (1.0f0 + 2.40f0 * (chi * chi * chi) / one_minus_chi)
    j_max_eff = j0_ref * (0.45f0 + 0.75f0 * f_cond) * f_quin * (0.30f0 + 0.90f0 * f_cata) * (0.55f0 + 0.50f0 * eps_0)
    tau_rc_eff = base.tau_rc * (0.75f0 + 0.55f0 * eps_eff) * f_quin

    cp = CoreParams(
        mode,
        base.E0_V,
        j_max_eff,
        base.alpha,
        K_half_eff,
        R_ohm_eff,
        k_para_eff,
        base.n_e,
        tau_rc_eff,
        base.v_ratio,
        base.theta_cap
    )
    pb = PyrolysisBiocharParams(
        T_pyro_C, eps_0, L_f_um, chi, eps_eff,
        R_char, R_ohm_eff, K_half_eff, j_max_eff, k_para_eff
    )
    return cp, pb
end

"""
step_core(c_bulk, theta, eta, q_cum, c_para_loss, E_app, dt, p)

Advances the reduced-order electrochemical state by `dt` seconds with exact
mass conservation (`C_phys = (c0 - c_bulk) - v_ratio*(q_cum/n_e + c_para_loss)`).
"""
@inline function step_core(
    c_bulk::Float32,
    theta::Float32,
    eta::Float32,
    q_cum::Float32,
    c_para_loss::Float32,
    E_app::Float32,
    dt::Float32,
    p::CoreParams
)
    c_pos = max(0.0f0, c_bulk)
    monod = c_pos / (p.K_half + c_pos + 1.0f-7)
    avail_site = clamp(1.0f0 - 0.35f0 * theta, 0.08f0, 1.0f0)

    j_target = 0.0f0
    j_para = 0.0f0
    if p.mode == MODE_BIOCHAR_MFC
        nernst_arg = clamp(p.alpha * FRT * eta, -14.0f0, 14.0f0)
        nernst_term = 1.0f0 / (1.0f0 + exp(-nernst_arg))
        j_target = p.j_scale * monod * nernst_term
        para_drive = clamp(4.5f0 * (eta - 0.28f0), -8.0f0, 8.0f0)
        j_para = p.k_para * monod * (0.60f0 + 0.40f0 / (1.0f0 + exp(-para_drive)))
    else
        nernst_arg = clamp(-p.alpha * FRT * eta, -14.0f0, 14.0f0)
        nernst_term = 1.0f0 / (1.0f0 + exp(-nernst_arg))
        j_target = p.j_scale * monod * nernst_term
        j_para = p.k_para * (0.40f0 + 0.60f0 * nernst_term)
    end

    max_equiv_rate = c_pos / max(dt * p.v_ratio, 1.0f-7)
    req_equiv_rate = (j_target / p.n_e) + (p.mode == MODE_BIOCHAR_MFC ? (j_para / p.n_e) : 0.0f0)
    if req_equiv_rate > max_equiv_rate && req_equiv_rate > 1.0f-9
        scale_lim = max_equiv_rate / req_equiv_rate
        j_target *= scale_lim
        if p.mode == MODE_BIOCHAR_MFC
            j_para *= scale_lim
        end
    end

    j_tot = j_target + j_para
    eta_F = j_tot > 1.0f-7 ? (j_target / j_tot) : 1.0f0

    sign_pol = (p.mode == MODE_BIOCHAR_MFC) ? 1.0f0 : -1.0f0
    ir_drop_V = 1.0f-3 * j_tot * p.R_ohm
    eta_ss = (E_app - p.E0_V) - sign_pol * ir_drop_V
    alpha_rc = dt / (p.tau_rc + dt)
    eta_next = eta + alpha_rc * (eta_ss - eta)

    dq_target = j_target * dt
    q_next = q_cum + dq_target
    para_stoich = (p.mode == MODE_BIOCHAR_MFC) ? (j_para * dt / p.n_e) : 0.0f0
    c_para_next = c_para_loss + para_stoich
    dc = p.v_ratio * ((dq_target / p.n_e) + para_stoich)
    c_next = max(0.0f0, c_pos - dc)
    theta_next = clamp(theta + dq_target / p.theta_cap, 0.0f0, 1.0f0)

    E_cell = (p.mode == MODE_BIOCHAR_MFC) ? max(0.0f0, 0.78f0 - E_app - ir_drop_V) : abs(E_app - p.E0_V)
    P_mw_cm2 = j_target * E_cell

    return c_next, theta_next, eta_next, q_next, c_para_next, j_target, j_tot, eta_F, P_mw_cm2
end

@inline function simulate_trajectory(
    p::CoreParams,
    E_app::Float32;
    c0::Float32 = 1.0f0,
    dt::Float32 = 0.5f0,
    n_steps::Int32 = Int32(240)
)
    c_bulk = c0
    theta = 0.02f0
    eta = 0.0f0
    q_cum = 0.0f0
    c_para = 0.0f0
    sum_j_target = 0.0f0
    sum_j_tot = 0.0f0
    sum_P = 0.0f0

    i = Int32(1)
    while i <= n_steps
        c_bulk, theta, eta, q_cum, c_para, j_t, j_tot, _, P_mw = step_core(
            c_bulk, theta, eta, q_cum, c_para, E_app, dt, p
        )
        sum_j_target += j_t
        sum_j_tot += j_tot
        sum_P += P_mw
        i += Int32(1)
    end

    inv_n = 1.0f0 / Float32(n_steps)
    mean_j_t = sum_j_target * inv_n
    mean_eta_F = sum_j_tot > 1.0f-6 ? (sum_j_target / sum_j_tot) : 1.0f0
    mean_P_mw = sum_P * inv_n
    yield_frac = (c0 - c_bulk) / max(c0, 1.0f-6)
    accounted_c = p.v_ratio * ((q_cum / p.n_e) + c_para)
    C_phys = abs((c0 - c_bulk) - accounted_c)
    return c_bulk, theta, eta, q_cum, mean_j_t, mean_eta_F, mean_P_mw, yield_frac, C_phys
end

function edge_weight_sensitivities(
    T_pyro_C::Float32,
    eps_0::Float32,
    L_f_um::Float32,
    E_app::Float32;
    base_j_scale::Float32 = -1.0f0,
    base_K_half::Float32 = -1.0f0,
    base_k_para::Float32 = -1.0f0
)::EdgeSensitivities
    _, pb_T_hi = biochar_consolidated_params(
        T_pyro_C + 5.0f0, eps_0, L_f_um;
        base_j_scale = base_j_scale, base_K_half = base_K_half, base_k_para = base_k_para
    )
    _, pb_T_lo = biochar_consolidated_params(
        T_pyro_C - 5.0f0, eps_0, L_f_um;
        base_j_scale = base_j_scale, base_K_half = base_K_half, base_k_para = base_k_para
    )
    dRohm_dT = (pb_T_hi.R_ohm_eff - pb_T_lo.R_ohm_eff) / 10.0f0

    dL = 1.5f0
    cp_L_hi, pb_L_hi = biochar_consolidated_params(
        T_pyro_C, eps_0, L_f_um + dL;
        base_j_scale = base_j_scale, base_K_half = base_K_half, base_k_para = base_k_para
    )
    cp_L_lo, pb_L_lo = biochar_consolidated_params(
        T_pyro_C, eps_0, max(2.0f0, L_f_um - dL);
        base_j_scale = base_j_scale, base_K_half = base_K_half, base_k_para = base_k_para
    )
    _, _, _, _, _, etaF_hi, P_hi, _, _ = simulate_trajectory(cp_L_hi, E_app)
    _, _, _, _, _, etaF_lo, P_lo, _, _ = simulate_trajectory(cp_L_lo, E_app)

    dchi = max(1.0f-5, pb_L_hi.chi - pb_L_lo.chi)
    dKhalf_dchi = (pb_L_hi.K_half_eff - pb_L_lo.K_half_eff) / dchi
    dP_dLf = (P_hi - P_lo) / ((L_f_um + dL) - max(2.0f0, L_f_um - dL))
    detaF_dchi = (etaF_hi - etaF_lo) / dchi

    return EdgeSensitivities(dRohm_dT, dKhalf_dchi, dP_dLf, detaF_dchi)
end

function selftest()
    test_cases = (
        (MODE_BIOCHAR_MFC, -0.08f0),
        (MODE_BIO_H2_MEC, -0.72f0),
    )
    for (m, e_app) in test_cases
        p = default_params(m)
        _, _, _, _, j_t, eta_F, _, y_f, c_err = simulate_trajectory(p, e_app)
        @assert c_err < 1.0f-4 "Mass balance violation in mode $m: C_phys=$c_err"
        @assert j_t > 0.01f0 "Zero target current in mode $m"
        @assert 0.20f0 < eta_F <= 1.0f0 "Out-of-range eta_F=$eta_F in mode $m"
        @assert y_f > 0.005f0 "Negligible conversion in mode $m"
    end

    cp_thin, pb_thin = biochar_consolidated_params(680.0f0, 0.68f0, 8.0f0)
    cp_opt,  pb_opt  = biochar_consolidated_params(680.0f0, 0.68f0, 48.0f0)
    cp_clog, pb_clog = biochar_consolidated_params(680.0f0, 0.68f0, 125.0f0)
    _, _, _, _, _, _, P_thin, _, err1 = simulate_trajectory(cp_thin, -0.06f0)
    _, _, _, _, _, _, P_opt,  _, err2 = simulate_trajectory(cp_opt,  -0.06f0)
    _, _, _, _, _, _, P_clog, _, err3 = simulate_trajectory(cp_clog, -0.06f0)

    @assert pb_thin.chi < pb_opt.chi < pb_clog.chi "Consolidation chi must increase with L_f"
    @assert pb_opt.R_ohm_eff < pb_clog.R_ohm_eff "Throat necking must raise R_ohm_eff at high chi"
    @assert P_opt > P_thin && P_opt > P_clog "Rivaling turnover failed: P_thin=$P_thin, P_opt=$P_opt, P_clog=$P_clog"
    @assert max(err1, max(err2, err3)) < 1.0f-4 "Consolidated biochar mass-balance drift"

    cp_cal, _ = biochar_consolidated_params(
        680.0f0, 0.68f0, 48.0f0;
        base_j_scale = 1.10f0, base_K_half = 0.12f0, base_k_para = 0.05f0
    )
    @assert cp_cal.j_scale > cp_opt.j_scale "Calibrated base_j_scale override failed"

    es = edge_weight_sensitivities(650.0f0, 0.68f0, 48.0f0, -0.06f0)
    @assert es.dRohm_dTpyro < 0.0f0 "Higher T_pyro must reduce solid-matrix R_ohm"
    @assert es.dKhalf_dchi > 0.0f0 "Consolidation chi must increase apparent K_half"

    println("ELECTROBIOCORE_JL_SELFTEST_PASS")
    return true
end

end # module CoreTransport
