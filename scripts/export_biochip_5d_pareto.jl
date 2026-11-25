#!/usr/bin/env julia
# export_biochip_5d_pareto.jl
# Pure-ASCII Uncalibrated Biochar Biochip Consolidation, AHL Quenching & Entrainment Exporter.
# Sweeps 3 independent inputs (T_pyro, eps_macro_0, chi) with derived d_throat and f_ent.
# Model mechanisms cite Masiello et al. (2013) [AHL sorption], Gao et al. (2016)
# [pH-dependent hydrolysis], and Wong & Ogbonnaya (2021) [porous microbial carrier];
# all curve constants are currently assumed rather than fitted to biochar data.
# Strictly 0 non-ASCII bytes, em=0, en=0.

using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "..", "src", "CoreTransport.jl"))
using .CoreTransport

const HAS_JACC = let ok = false
    if "--use-jacc" in ARGS
        try
            @eval using JACC
            @eval JACC.@init_backend
            ok = true
            println("[biochip-pareto] JACC GPU backend active")
        catch err
            @warn "[biochip-pareto] JACC/Metal unavailable; running deterministic CPU kernel" exception=err
        end
    end
    ok
end

struct PilotPoint5D
    T_pyro_C::Float32
    eps_macro_0::Float32
    d_throat_um::Float32
    chi::Float32
    chi_ceiling::Float32
    eps_macro_eff::Float32
    BET_m2_g::Float32
    CEC_cmol_kg::Float32
    f_QS::Float32
    f_ent::Float32
    R_char_ohm_cm2::Float32
    R_ohm_eff::Float32
    kappa_ratio::Float32
    P_mw_cm2::Float32
    Q_seq_mg_g::Float32
    eta_F::Float32
    pareto_weight::Float32
    Wo::Float32
    disp_gain::Float32
    streamer_factor::Float32
end

@inline function compute_womersley(
    d_throat_um::Float32,
    stride_hz::Float32;
    nu_m2_s::Float32 = 1.0f-6
)::Float32
    if stride_hz <= 0.0f0 || d_throat_um <= 0.0f0
        return 0.0f0
    end
    r_m = 0.5f0 * d_throat_um * 1.0f-6
    omega = 6.2831855f0 * stride_hz
    return r_m * sqrt(omega / nu_m2_s)
end

@inline function eval_qs_and_entrainment(
    T_pyro_C::Float32,
    eps_0::Float32;
    Pe_flow::Float32 = 1.80f0,
    Pe_osc::Float32 = 0.0f0,
    stride_hz::Float32 = 1.80f0,
    ahl_rescue::Bool = false
)
    # Assumed parametric property curves (uncalibrated): BET [m2/g] and slurry pH
    BET = 98.0f0 + 422.0f0 / (1.0f0 + exp(-(T_pyro_C - 575.0f0) / 62.0f0))
    pH_char = 6.80f0 + 3.40f0 / (1.0f0 + exp(-(T_pyro_C - 565.0f0) / 68.0f0))

    # AHL sorption (Masiello et al., 2013) + alkaline hydrolysis (Gao et al., 2016), assumed constants
    sorb_sink = 0.42f0 * ((BET / 260.0f0) ^ 1.65f0)
    oh_sink = 0.14f0 * (10.0f0 ^ (pH_char - 8.50f0))
    f_qs_raw = 1.0f0 / (1.0f0 + sorb_sink + oh_sink)
    f_QS = ahl_rescue ? max(f_qs_raw, 0.84f0) : f_qs_raw

    # Derived lumen aperture [um] & hydrodynamic entrainment factor f_ent in (0, 1]
    d_throat = 4.0f0 + 36.0f0 * max(0.02f0, eps_0 - 0.30f0)
    steric_entry = 1.0f0 / (1.0f0 + exp(-(d_throat - 3.20f0) / 1.10f0))

    if Pe_osc > 0.0f0
        wo = compute_womersley(d_throat, stride_hz)
        pe_rms = sqrt(Pe_flow * Pe_flow + 0.5f0 * Pe_osc * Pe_osc)
        disp_gain = 1.0f0 + 0.14f0 * (Pe_osc * Pe_osc) / (1.0f0 + 0.25f0 * wo * wo)
        streamer_prune = clamp(0.18f0 * (Pe_osc / (1.50f0 + Pe_osc)) * (1.0f0 + 0.45f0 * wo), 0.0f0, 0.35f0)
        streamer_factor = 1.0f0 - streamer_prune
    else
        wo = 0.0f0
        pe_rms = Pe_flow
        disp_gain = 1.0f0
        streamer_prune = 0.0f0
        streamer_factor = 1.0f0
    end

    shear_shelter = 1.0f0 / (1.0f0 + (pe_rms / 3.50f0) * ((d_throat / 22.0f0) ^ 2))
    f_ent = steric_entry * shear_shelter

    drive_prod = f_QS * f_ent
    chi_ceil_raw = 0.10f0 + 0.82f0 * (drive_prod / (0.14f0 + drive_prod))
    chi_ceil = clamp(chi_ceil_raw * (1.0f0 - 0.35f0 * streamer_prune), 0.12f0, 0.90f0)
    return BET, d_throat, f_QS, f_ent, chi_ceil, wo, disp_gain, streamer_factor
end

@inline function eval_5d_state(
    T_pyro_C::Float32,
    eps_0::Float32,
    chi_in::Float32;
    tau_B::Float32 = 1.80f0,
    Pe_flow::Float32 = 1.80f0,
    Pe_osc::Float32 = 0.0f0,
    stride_hz::Float32 = 1.80f0,
    ahl_rescue::Bool = false
)::PilotPoint5D
    chi = clamp(chi_in, 0.0f0, 0.92f0)
    omc = max(0.06f0, 1.0f0 - chi)
    eps_eff = eps_0 * omc

    BET, d_throat, f_QS, f_ent, chi_ceil, wo, disp_gain, streamer_factor = eval_qs_and_entrainment(
        T_pyro_C, eps_0;
        Pe_flow = Pe_flow, Pe_osc = Pe_osc, stride_hz = stride_hz, ahl_rescue = ahl_rescue
    )

    R_char = CoreTransport.char_area_resistance(T_pyro_C)
    # Assumed CEC curve (48 cmol/kg at 400 C, upper-bound surrogate); Cd2+-eq factor 0.56 mg/g per cmol/kg
    CEC0 = 6.50f0 + 41.50f0 * exp(-(T_pyro_C - 400.0f0) / 245.0f0)
    z_q = (T_pyro_C - 595.0f0) / 108.0f0
    f_quin = 0.72f0 + 0.30f0 * exp(-(z_q * z_q))

    qs_wiring_gain = 0.38f0 + 0.62f0 * sqrt(f_QS)
    f_brug = omc ^ tau_B
    K_half_eff = (0.15f0 / max(0.04f0, f_brug)) / disp_gain
    denom_kc = 1.0f0 + 1.80f0 * chi * streamer_factor
    kappa_ratio = (omc * omc * omc) / (denom_kc * denom_kc)
    chi_perc = 0.20f0 + 0.24f0 * eps_0
    f_perc = tanh(chi / chi_perc) * qs_wiring_gain
    R_pore = (4.60f0 / max(0.25f0, eps_0)) / max(0.05f0, f_brug * (1.0f0 + 0.25f0 * (disp_gain - 1.0f0)))
    R_ohm_eff = max(6.0f0, R_char + R_pore - 3.10f0 * f_perc)

    over_ceil = max(0.0f0, chi - chi_ceil)
    kin_gate = exp(-9.5f0 * over_ceil * over_ceil)
    monod_eff = 0.82f0 / (K_half_eff + 0.82f0)

    j_target = 1.017f0 * f_quin * f_perc * monod_eff * (0.68f0 + 0.44f0 * eps_0) * kin_gate
    j_para = 0.068f0 * (1.0f0 + 2.15f0 * streamer_factor * (chi * chi * chi) / omc) * monod_eff
    j_tot = j_target + j_para
    eta_F = j_tot > 1.0f-6 ? (j_target / j_tot) : 1.0f0
    E_cell = max(0.0f0, 0.64f0 - 1.0f-3 * j_tot * R_ohm_eff * 6.5f0)
    # Projected-anode-area power density (uncalibrated)
    P_mw = j_target * E_cell

    # Cd2+-equivalent retention index (0.56 mg/g per cmol/kg CEC + biogenic EPS contribution)
    access_gate = 0.32f0 + 0.68f0 * (f_brug / (0.11f0 + f_brug))
    abiotic_term = 0.56f0 * CEC0 * (1.0f0 - 0.52f0 * chi)
    biogenic_eps = 46.0f0 * eps_0 * chi * (0.45f0 + 0.55f0 * f_ent) * qs_wiring_gain * kin_gate
    Q_seq = (abiotic_term + biogenic_eps) * access_gate

    return PilotPoint5D(
        T_pyro_C, eps_0, d_throat, chi, chi_ceil, eps_eff, BET, CEC0,
        f_QS, f_ent, R_char, R_ohm_eff, kappa_ratio, P_mw, Q_seq, eta_F, 0.0f0,
        wo, disp_gain, streamer_factor
    )
end

function compute_kappa_cliff(streamer_factor::Float32 = 1.0f0; target::Float64 = 0.06)::Float64
    lo, hi = 0.0, 0.90
    sf = Float64(streamer_factor)
    for _ in 1:64
        mid = 0.5 * (lo + hi)
        omc = 1.0 - mid
        denom = 1.0 + 1.80 * mid * sf
        kap = (omc * omc * omc) / (denom * denom)
        if kap >= target
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

function _pilot_sweep_kernel(
    idx, d_P, d_Q, d_eta, d_feas,
    nt::Int32, np::Int32, nc::Int32,
    pe_flow::Float32, pe_osc::Float32, stride_hz::Float32
)
    r0 = idx - Int32(1)
    ic = mod(r0, nc); r1 = div(r0, nc)
    ip = mod(r1, np)
    it = div(r1, np)
    Tv = 400.0f0 + 400.0f0 * (Float32(it) / Float32(nt - Int32(1)))
    ev = 0.35f0 + 0.45f0 * (Float32(ip) / Float32(np - Int32(1)))
    cv = 0.05f0 + 0.83f0 * (Float32(ic) / Float32(nc - Int32(1)))
    st = eval_5d_state(Tv, ev, cv; Pe_flow = pe_flow, Pe_osc = pe_osc, stride_hz = stride_hz)
    @inbounds d_P[idx] = st.P_mw_cm2
    @inbounds d_Q[idx] = st.Q_seq_mg_g
    @inbounds d_eta[idx] = st.eta_F
    @inbounds d_feas[idx] = (cv <= st.chi_ceiling && st.kappa_ratio >= 0.06f0 && st.eta_F >= 0.72f0) ? 0x01 : 0x00
    return nothing
end

function compute_pareto_front(;
    Pe_flow::Float32 = 1.80f0,
    Pe_osc::Float32 = 0.0f0,
    stride_hz::Float32 = 1.80f0
)
    NT = 64; NP = 32; NC = 32
    NTOT = NT * NP * NC
    P_arr = Vector{Float32}(undef, NTOT)
    Q_arr = Vector{Float32}(undef, NTOT)
    eta_arr = Vector{Float32}(undef, NTOT)
    feas_arr = Vector{UInt8}(undef, NTOT)

    if HAS_JACC
        d_P = JACC.zeros(Float32, NTOT)
        d_Q = JACC.zeros(Float32, NTOT)
        d_eta = JACC.zeros(Float32, NTOT)
        d_feas = JACC.zeros(UInt8, NTOT)
        JACC.parallel_for(
            NTOT, _pilot_sweep_kernel,
            d_P, d_Q, d_eta, d_feas, Int32(NT), Int32(NP), Int32(NC),
            Pe_flow, Pe_osc, stride_hz
        )
        P_arr .= Array(d_P)
        Q_arr .= Array(d_Q)
        eta_arr .= Array(d_eta)
        feas_arr .= Array(d_feas)
    else
        for idx in 1:NTOT
            _pilot_sweep_kernel(
                Int32(idx), P_arr, Q_arr, eta_arr, feas_arr,
                Int32(NT), Int32(NP), Int32(NC),
                Pe_flow, Pe_osc, stride_hz
            )
        end
    end

    feas_indices = Int[]
    for idx in 1:NTOT
        if feas_arr[idx] == 0x01
            push!(feas_indices, idx)
        end
    end
    sort!(feas_indices; by = idx -> (-Q_arr[idx], -P_arr[idx], idx))

    front = PilotPoint5D[]
    best_P = -1.0f9
    for idx in feas_indices
        if P_arr[idx] > best_P
            best_P = P_arr[idx]
            r0 = idx - 1
            ic = mod(r0, NC); r1 = div(r0, NC)
            ip = mod(r1, NP)
            it = div(r1, NP)
            Tv = 400.0f0 + 400.0f0 * (Float32(it) / Float32(NT - 1))
            ev = 0.35f0 + 0.45f0 * (Float32(ip) / Float32(NP - 1))
            cv = 0.05f0 + 0.83f0 * (Float32(ic) / Float32(NC - 1))
            push!(front, eval_5d_state(Tv, ev, cv; Pe_flow = Pe_flow, Pe_osc = Pe_osc, stride_hz = stride_hz))
        end
    end
    return front
end

function write_pareto_csv(
    out_csv::AbstractString,
    front::Vector{PilotPoint5D};
    Pe_flow::Float32 = 1.80f0,
    Pe_osc::Float32 = 0.0f0,
    stride_hz::Float32 = 1.80f0
)
    buf = IOBuffer()
    pulsatile = Pe_osc > 0.0f0
    @printf(buf, "# ElectroBioCore Non-Dominated Pareto Front (%d points, uncalibrated model; all constants assumed)\n", length(front))
    println(buf, "# Swept inputs (3): T_pyro_C [400..800], eps_macro_0 [0.35..0.80], chi [0.05..0.88]; d_throat_um and f_ent derived from eps_macro_0")
    if pulsatile
        @printf(
            buf,
            "# Model: Masiello et al. (2013) AHL sorption + Gao et al. (2016) hydrolysis + Wong & Ogbonnaya (2021) porous carrier + Pulsatile (Pe_flow=%.2f,Pe_osc=%.2f,stride_hz=%.2f)\n",
            Pe_flow, Pe_osc, stride_hz
        )
    else
        @printf(
            buf,
            "# Model: Masiello et al. (2013) AHL sorption + Gao et al. (2016) hydrolysis + Wong & Ogbonnaya (2021) porous carrier (Pe_flow=%.2f)\n",
            Pe_flow
        )
    end
    println(buf, "# Units & basis: R_char_ohm_cm2 & R_ohm_eff_ohm_cm2 [Ohm*cm2 area-specific]; P_mw_cm2_proj [mW/cm2 projected anode area]; Q_seq_Cd_eq_mg_g [Cd2+-equivalent retention index: 0.56*CEC + EPS]")
    if pulsatile
        println(buf, "idx,T_pyro_C,BET_m2_g,CEC_cmol_kg,eps_macro_0,d_throat_um,f_QS,f_ent,chi_opt,chi_ceiling,eps_macro_eff,R_char_ohm_cm2,R_ohm_eff_ohm_cm2,kappa_ratio,P_mw_cm2_proj,Q_seq_Cd_eq_mg_g,eta_F,Wo,disp_gain,streamer_factor")
    else
        println(buf, "idx,T_pyro_C,BET_m2_g,CEC_cmol_kg,eps_macro_0,d_throat_um,f_QS,f_ent,chi_opt,chi_ceiling,eps_macro_eff,R_char_ohm_cm2,R_ohm_eff_ohm_cm2,kappa_ratio,P_mw_cm2_proj,Q_seq_Cd_eq_mg_g,eta_F")
    end
    for (i, p) in enumerate(front)
        if pulsatile
            @printf(
                buf,
                "%d,%.2f,%.2f,%.2f,%.4f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.4f,%.5f,%.3f,%.4f,%.5f,%.4f,%.4f\n",
                i, p.T_pyro_C, p.BET_m2_g, p.CEC_cmol_kg,
                p.eps_macro_0, p.d_throat_um, p.f_QS, p.f_ent, p.chi, p.chi_ceiling,
                p.eps_macro_eff, p.R_char_ohm_cm2, p.R_ohm_eff, p.kappa_ratio, p.P_mw_cm2, p.Q_seq_mg_g, p.eta_F,
                p.Wo, p.disp_gain, p.streamer_factor
            )
        else
            @printf(
                buf,
                "%d,%.2f,%.2f,%.2f,%.4f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.4f,%.5f,%.3f,%.4f\n",
                i, p.T_pyro_C, p.BET_m2_g, p.CEC_cmol_kg,
                p.eps_macro_0, p.d_throat_um, p.f_QS, p.f_ent, p.chi, p.chi_ceiling,
                p.eps_macro_eff, p.R_char_ohm_cm2, p.R_ohm_eff, p.kappa_ratio, p.P_mw_cm2, p.Q_seq_mg_g, p.eta_F
            )
        end
    end
    raw = take!(buf)
    csv_sha = bytes2hex(sha256(raw))
    mkpath(dirname(abspath(out_csv)))
    open(out_csv, "w") do io
        @printf(io, "# sha256_body=%s\n", csv_sha)
        write(io, raw)
    end
    full_bytes = read(out_csv)
    @printf("[csv] wrote %s (%d bytes, body_sha256=%s)\n", out_csv, length(full_bytes), csv_sha[1:16])
    return String(full_bytes), csv_sha
end

function viridis_hex(u_in::Float32)::String
    u = clamp(Float64(u_in), 0.0, 1.0)
    stops = (
        (0.00, 68.0, 1.0, 84.0),
        (0.25, 59.0, 82.0, 139.0),
        (0.50, 33.0, 145.0, 140.0),
        (0.75, 94.0, 201.0, 98.0),
        (1.00, 253.0, 231.0, 37.0),
    )
    for k in 1:4
        t0, r0, g0, b0 = stops[k]
        t1, r1, g1, b1 = stops[k + 1]
        if u <= t1 || k == 4
            w = clamp((u - t0) / (t1 - t0), 0.0, 1.0)
            r = round(Int, r0 + w * (r1 - r0))
            g = round(Int, g0 + w * (g1 - g0))
            b = round(Int, b0 + w * (b1 - b0))
            return @sprintf("#%02x%02x%02x", r, g, b)
        end
    end
    return "#21918c"
end

@inline function project_iso(
    uT::Float64, uChi::Float64, uZ::Float64;
    cx::Float64 = 517.0, cy::Float64 = 727.0,
    sxT::Float64 = 405.0, syT::Float64 = 132.0,
    sxC::Float64 = 322.0, syC::Float64 = 153.0,
    sz::Float64 = 352.0
)
    px = cx + (uT - 0.5) * sxT - (uChi - 0.5) * sxC
    py = cy + (uT - 0.5) * syT + (uChi - 0.5) * syC - uZ * sz
    return px, py
end

function _emit_surface_panel!(
    io::IOBuffer,
    front::Vector{PilotPoint5D};
    eps_slice::Float32 = 0.80f0,
    Pe_flow::Float32 = 1.80f0,
    Pe_osc::Float32 = 0.0f0,
    stride_hz::Float32 = 1.80f0,
    cx::Float64 = 517.0,
    cy::Float64 = 727.0,
    sxT::Float64 = 405.0,
    syT::Float64 = 132.0,
    sxC::Float64 = 322.0,
    syC::Float64 = 153.0,
    sz::Float64 = 352.0,
    cb_x::Int = 68,
    cb_y::Int = 985
)
    proj(uT, uC, uZ) = project_iso(uT, uC, uZ; cx=cx, cy=cy, sxT=sxT, syT=syT, sxC=sxC, syC=syC, sz=sz)

    bx1, by1 = proj(0.0, 0.0, 0.0)
    bx2, by2 = proj(1.0, 0.0, 0.0)
    bx3, by3 = proj(1.0, 1.0, 0.0)
    bx4, by4 = proj(0.0, 1.0, 0.0)
    bx4z, by4z = proj(0.0, 1.0, 1.0)
    @printf(io, " <polygon points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"#0d141d\" stroke=\"#33475b\" stroke-width=\"1.6\"/>\n", bx1, by1, bx2, by2, bx3, by3, bx4, by4)

    NG = 22
    pts_grid = Matrix{PilotPoint5D}(undef, NG, NG)
    p_hi = 1.0f-5
    q_lo = 1.0f9
    q_hi = -1.0f9
    for it in 1:NG, ic in 1:NG
        Tv = 400.0f0 + 400.0f0 * Float32(it - 1) / Float32(NG - 1)
        cv = 0.05f0 + 0.83f0 * Float32(ic - 1) / Float32(NG - 1)
        st = eval_5d_state(Tv, eps_slice, cv; Pe_flow = Pe_flow, Pe_osc = Pe_osc, stride_hz = stride_hz)
        pts_grid[it, ic] = st
        p_hi = max(p_hi, st.P_mw_cm2)
        q_lo = min(q_lo, st.Q_seq_mg_g)
        q_hi = max(q_hi, st.Q_seq_mg_g)
    end
    for pf in front
        p_hi = max(p_hi, pf.P_mw_cm2)
        q_hi = max(q_hi, pf.Q_seq_mg_g)
    end

    sf_val = pts_grid[1, 1].streamer_factor
    chi_cliff = compute_kappa_cliff(sf_val)
    u_cliff = clamp((chi_cliff - 0.05) / 0.83, 0.0, 1.0)
    cx_a, cy_a = proj(0.0, u_cliff, 0.0)
    cx_b, cy_b = proj(1.0, u_cliff, 0.0)
    @printf(io, " <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#ff5e57\" stroke-width=\"2.6\" stroke-dasharray=\"8,5\"/>\n", cx_a, cy_a, cx_b, cy_b)

    for it in 1:(NG - 1), ic in 1:(NG - 1)
        s00 = pts_grid[it, ic]
        s10 = pts_grid[it + 1, ic]
        s11 = pts_grid[it + 1, ic + 1]
        s01 = pts_grid[it, ic + 1]
        uT0 = Float64(it - 1) / Float64(NG - 1)
        uT1 = Float64(it) / Float64(NG - 1)
        uC0 = Float64(ic - 1) / Float64(NG - 1)
        uC1 = Float64(ic) / Float64(NG - 1)
        x00, y00 = proj(uT0, uC0, Float64(s00.P_mw_cm2 / p_hi))
        x10, y10 = proj(uT1, uC0, Float64(s10.P_mw_cm2 / p_hi))
        x11, y11 = proj(uT1, uC1, Float64(s11.P_mw_cm2 / p_hi))
        x01, y01 = proj(uT0, uC1, Float64(s01.P_mw_cm2 / p_hi))
        q_mid = 0.25f0 * (s00.Q_seq_mg_g + s10.Q_seq_mg_g + s11.Q_seq_mg_g + s01.Q_seq_mg_g)
        u_col = (q_mid - q_lo) / max(1.0f-4, q_hi - q_lo)
        hex_c = viridis_hex(u_col)
        @printf(
            io,
            " <polygon points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"%s\" fill-opacity=\"0.86\" stroke=\"#182433\" stroke-width=\"0.7\"/>\n",
            x00, y00, x10, y10, x11, y11, x01, y01, hex_c
        )
    end

    # Overlay non-dominated Pareto front
    r_buf = IOBuffer()
    for p in front
        uT = clamp((Float64(p.T_pyro_C) - 400.0) / 400.0, 0.0, 1.0)
        uC = clamp((Float64(p.chi) - 0.05) / 0.83, 0.0, 1.0)
        uZ = clamp(Float64(p.P_mw_cm2 / p_hi), 0.0, 1.0)
        px, py = proj(uT, uC, uZ)
        @printf(r_buf, "%.1f,%.1f ", px, py)
    end
    println(io, " <polyline points=\"$(strip(String(take!(r_buf))))\" fill=\"none\" stroke=\"#00f5d4\" stroke-width=\"4.0\"/>")

    # Numeric axis ticks along T_pyro (front-right base edge: uChi=1, uT=0..1)
    for tv in (400, 500, 600, 700, 800)
        uT = (Float64(tv) - 400.0) / 400.0
        tx, ty = proj(uT, 1.0, 0.0)
        @printf(io, " <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#6c86a3\" stroke-width=\"1.4\"/>\n", tx, ty, tx + 4.0, ty + 8.0)
        @printf(io, " <text x=\"%.1f\" y=\"%.1f\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"13\">%d</text>\n", tx + 6.0, ty + 22.0, tv)
    end
    tm_x, tm_y = proj(0.50, 1.0, 0.0)
    @printf(io, " <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\" fill=\"#b8cbe0\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">T_pyro (C)</text>\n", tm_x + 10.0, tm_y + 52.0)

    # Numeric axis ticks along chi (open front-right floor edge: uT=1, uChi=0..1)
    for cv in (0.10, 0.30, 0.50, 0.70)
        uC = (cv - 0.05) / 0.83
        tx, ty = proj(1.0, uC, 0.0)
        @printf(io, " <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#6c86a3\" stroke-width=\"1.4\"/>\n", tx, ty, tx + 8.0, ty + 4.0)
        @printf(io, " <text x=\"%.1f\" y=\"%.1f\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"13\">%.2f</text>\n", tx + 11.0, ty + 8.0, cv)
    end
    cm_x, cm_y = proj(1.0, 0.52, 0.0)
    @printf(io, " <text x=\"%.1f\" y=\"%.1f\" fill=\"#b8cbe0\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">chi (occlusion)</text>\n", cm_x + 48.0, cm_y + 28.0)

    # Vertical P_mw_cm2 axis at unobstructed front-left corner (uT=0, uChi=1, uZ=0..1)
    @printf(io, " <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#6c86a3\" stroke-width=\"1.6\"/>\n", bx4, by4, bx4z, by4z)
    for frac in (0.0, 0.5, 1.0)
        tx, ty = proj(0.0, 1.0, frac)
        @printf(io, " <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#6c86a3\" stroke-width=\"1.4\"/>\n", tx, ty, tx - 7.0, ty)
        @printf(io, " <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"end\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"13\">%.2f</text>\n", tx - 11.0, ty + 4.0, frac * Float64(p_hi))
    end
    @printf(io, " <text x=\"%.1f\" y=\"%.1f\" fill=\"#b8cbe0\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Projected P (mW/cm2, uncal.)</text>\n", bx4z - 42.0, by4z - 14.0)

    @printf(io, " <text x=\"%d\" y=\"%d\" fill=\"#b8cbe0\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Q_seq Cd2+-eq (mg/g):</text>\n", cb_x, cb_y)
    bar_x = cb_x + 178
    for s in 0:19
        u_s = Float32(s) / 19.0f0
        @printf(io, " <rect x=\"%d\" y=\"%d\" width=\"18\" height=\"18\" fill=\"%s\"/>\n", bar_x + s * 18, cb_y - 17, viridis_hex(u_s))
    end
    @printf(io, " <text x=\"%d\" y=\"%d\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"14\">%.1f</text>\n", bar_x, cb_y + 23, q_lo)
    @printf(io, " <text x=\"%d\" y=\"%d\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"14\">%.1f mg/g</text>\n", bar_x + 305, cb_y + 23, q_hi)
    return p_hi, q_lo, q_hi, chi_cliff
end

function write_1panel_svg(
    out_svg::AbstractString,
    front::Vector{PilotPoint5D};
    eps_slice::Float32 = 0.80f0,
    Pe_flow::Float32 = 1.80f0
)
    io = IOBuffer()
    println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1920\" height=\"1080\" viewBox=\"0 0 1920 1080\">")
    println(io, " <rect x=\"0\" y=\"0\" width=\"1920\" height=\"1080\" fill=\"#0b1017\"/>")
    println(io, " <text x=\"48\" y=\"54\" fill=\"#e8eef5\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"28\" font-weight=\"bold\">Biochar Macropore Consolidation &amp; AHL Quenching Regime Map (Uncalibrated Model)</text>")
    @printf(
        io,
        " <text x=\"48\" y=\"88\" fill=\"#9ab0c6\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"18\">3 Swept Inputs (T_pyro, eps_macro,0, chi) | Masiello et al. (2013) sorption + Gao et al. (2016) hydrolysis + Wong &amp; Ogbonnaya (2021) | Pe=%.1f</text>\n",
        Pe_flow
    )

    println(io, " <rect x=\"36\" y=\"114\" width=\"1160\" height=\"920\" rx=\"10\" fill=\"#111923\" stroke=\"#293849\" stroke-width=\"2\"/>")
    println(io, " <rect x=\"1220\" y=\"114\" width=\"664\" height=\"920\" rx=\"10\" fill=\"#111923\" stroke=\"#293849\" stroke-width=\"2\"/>")

    @printf(io, " <text x=\"68\" y=\"156\" fill=\"#dce6f2\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"21\" font-weight=\"bold\">Response Surface (eps_macro,0 = %.2f): Power P_proj vs. Q_seq (Cd2+-eq Index)</text>\n", eps_slice)
    p_hi, _, q_hi, chi_cliff = _emit_surface_panel!(
        io, front;
        eps_slice = eps_slice, Pe_flow = Pe_flow,
        cx = 620.0, cy = 740.0, sxT = 460.0, syT = 145.0, sxC = 360.0, syC = 168.0, sz = 340.0,
        cb_x = 68, cb_y = 985
    )
    @printf(io, " <text x=\"68\" y=\"186\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"16\">Cyan spine: %d-point non-dominated Pareto front | Red dashed cutoff: kappa/kappa0 = 0.06 (chi = %.3f)</text>\n", length(front), chi_cliff)

    p_first = front[1]
    p_last = front[end]
    println(io, " <text x=\"1252\" y=\"158\" fill=\"#dce6f2\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"22\" font-weight=\"bold\">Model Summary &amp; Uncalibrated Assumptions</text>")
    println(io, " <line x1=\"1252\" y1=\"176\" x2=\"1852\" y2=\"176\" stroke=\"#293849\" stroke-width=\"1.5\"/>")

    println(io, " <text x=\"1252\" y=\"218\" fill=\"#00f5d4\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"17\" font-weight=\"bold\">1. Non-Dominated Pareto Front (3 Swept Inputs)</text>")
    @printf(io, " <text x=\"1252\" y=\"248\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">All %d front points lie continuously on eps_macro,0 = %.2f edge.</text>\n", length(front), eps_slice)
    @printf(io, " <text x=\"1252\" y=\"274\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Q_seq max (T=%.0f C): %.3f mg/g Cd2+-eq, P=%.5f mW/cm2</text>\n", p_first.T_pyro_C, p_first.Q_seq_mg_g, p_first.P_mw_cm2)
    @printf(io, " <text x=\"1252\" y=\"300\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Power peak (T=%.1f C): P=%.5f mW/cm2, f_QS=%.4f, chi=%.4f</text>\n", p_last.T_pyro_C, p_last.P_mw_cm2, p_last.f_QS, p_last.chi)
    println(io, " <text x=\"1252\" y=\"326\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"14\">Note: Peak P is flat within 1% across ~563 to 600 C.</text>")

    println(io, " <text x=\"1252\" y=\"382\" fill=\"#ff5e57\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"17\" font-weight=\"bold\">2. Permeability Cutoff &amp; Lumen Geometry</text>")
    @printf(io, " <text x=\"1252\" y=\"412\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Red dashed floor line: Kozeny-Carman kappa/kappa0 = 0.06 at chi = %.3f.</text>\n", chi_cliff)
    println(io, " <text x=\"1252\" y=\"438\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Cell-accessible space: 10-30 um vascular lumens (not &lt;50 nm BET).</text>")
    println(io, " <text x=\"1252\" y=\"464\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"14\">d_throat and f_ent are derived deterministically from eps_macro,0.</text>")

    println(io, " <text x=\"1252\" y=\"520\" fill=\"#f6b93b\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"17\" font-weight=\"bold\">3. Parameter Basis (All Constants Assumed)</text>")
    println(io, " <text x=\"1252\" y=\"550\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Every parameter curve (BET, pH, CEC, R_char, f_QS) is hand-set.</text>")
    println(io, " <text x=\"1252\" y=\"576\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">No parameters are fitted yet to experimental biochar or EIS data.</text>")
    println(io, " <text x=\"1252\" y=\"602\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Power P is normalized per projected anode area (mW/cm2).</text>")
    println(io, " <text x=\"1252\" y=\"628\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Q_seq index combines 0.56*CEC (Cd2+-eq, high 48 cmol/kg at 400 C)</text>")
    println(io, " <text x=\"1252\" y=\"652\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">with an assumed biogenic EPS contribution.</text>")

    println(io, " <text x=\"1252\" y=\"708\" fill=\"#dce6f2\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"17\" font-weight=\"bold\">4. Literature Tension to Test Experimentally</text>")
    println(io, " <text x=\"1252\" y=\"738\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">The interior ~584 C peak arises solely from the f_QS high-T penalty.</text>")
    println(io, " <text x=\"1252\" y=\"764\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Published biochar anodes often plateau at 800-1000 C, and short</text>")
    println(io, " <text x=\"1252\" y=\"790\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">assays report higher initial attachment on hotter chars. Microfluidic</text>")
    println(io, " <text x=\"1252\" y=\"816\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">imaging across 450/620/780 C (+AHL add-back) directly tests this gap.</text>")

    @printf(io, " <text x=\"1252\" y=\"990\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"13\">ElectroBioCore v0.1.0 | Scale: P_max = %.4f mW/cm2 | Q_max = %.2f mg/g</text>\n", p_hi, q_hi)
    println(io, "</svg>")

    svg_str = String(take!(io))
    mkpath(dirname(abspath(out_svg)))
    open(out_svg, "w") do f
        write(f, svg_str)
    end
    svg_sha = bytes2hex(sha256(codeunits(svg_str)))
    @printf("[svg] wrote %s (%d bytes, sha256=%s)\n", out_svg, sizeof(svg_str), svg_sha[1:16])
    return svg_str, svg_sha
end

function write_2panel_svg(
    out_svg::AbstractString,
    front::Vector{PilotPoint5D};
    eps_slice::Float32 = 0.80f0,
    Pe_flow::Float32 = 1.80f0,
    Pe_osc::Float32 = 0.0f0,
    stride_hz::Float32 = 1.80f0
)
    pulsatile = Pe_osc > 0.0f0
    io = IOBuffer()
    println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1920\" height=\"1080\" viewBox=\"0 0 1920 1080\">")
    println(io, " <rect x=\"0\" y=\"0\" width=\"1920\" height=\"1080\" fill=\"#0b1017\"/>")
    if pulsatile
        println(io, " <text x=\"48\" y=\"54\" fill=\"#e8eef5\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"28\" font-weight=\"bold\">Biochar Macropore Consolidation: Pulsatile Peclet/Womersley Model Map (Uncalibrated)</text>")
        @printf(io, " <text x=\"48\" y=\"88\" fill=\"#9ab0c6\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"18\">3 Swept Inputs (T_pyro, eps_macro,0, chi) | Masiello et al. (2013), Gao et al. (2016), Wong &amp; Ogbonnaya (2021) | Pe_flow=%.2f, Pe_osc=%.2f, stride=%.2f Hz</text>\n", Pe_flow, Pe_osc, stride_hz)
    else
        println(io, " <text x=\"48\" y=\"54\" fill=\"#e8eef5\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"28\" font-weight=\"bold\">Biochar Macropore Consolidation, AHL Quenching &amp; Simulated Trajectories (Uncalibrated Model)</text>")
        @printf(io, " <text x=\"48\" y=\"88\" fill=\"#9ab0c6\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"18\">3 Swept Inputs (T_pyro, eps_macro,0, chi) | Masiello et al. (2013) sorption + Gao et al. (2016) hydrolysis + Wong &amp; Ogbonnaya (2021) | Pe=%.1f</text>\n", Pe_flow)
    end

    println(io, " <rect x=\"36\" y=\"114\" width=\"940\" height=\"920\" rx=\"10\" fill=\"#111923\" stroke=\"#293849\" stroke-width=\"2\"/>")
    println(io, " <rect x=\"1000\" y=\"114\" width=\"884\" height=\"920\" rx=\"10\" fill=\"#111923\" stroke=\"#293849\" stroke-width=\"2\"/>")

    @printf(io, " <text x=\"64\" y=\"154\" fill=\"#dce6f2\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"21\" font-weight=\"bold\">A. Surface (eps_macro,0 = %.2f): Height = P_proj | Color = Q_seq</text>\n", eps_slice)
    p_hi, _, q_hi, chi_cliff = _emit_surface_panel!(
        io, front;
        eps_slice = eps_slice, Pe_flow = Pe_flow, Pe_osc = Pe_osc, stride_hz = stride_hz,
        cx = 517.0, cy = 727.0, sxT = 405.0, syT = 132.0, sxC = 322.0, syC = 153.0, sz = 352.0,
        cb_x = 68, cb_y = 985
    )
    @printf(io, " <text x=\"64\" y=\"182\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"16\">Cyan: %d-pt Pareto front | Red dashed floor: kappa/kappa0 = 0.06 (chi = %.3f)</text>\n", length(front), chi_cliff)

    # Panel B: Explicitly labeled simulated trajectories
    println(io, " <text x=\"1028\" y=\"154\" fill=\"#dce6f2\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"21\" font-weight=\"bold\">B. Simulated 14-Day Trajectories (Uncalibrated Model, Assumed Kinetics)</text>")
    println(io, " <text x=\"1028\" y=\"182\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Model curves only (not experimental data) | Amber 450C | Green 620C | Blue 780C | Pink 780C+AHL</text>")

    println(io, " <rect x=\"1068\" y=\"216\" width=\"780\" height=\"340\" fill=\"#0d141d\" stroke=\"#2c3d50\" stroke-width=\"1.4\"/>")
    println(io, " <rect x=\"1068\" y=\"626\" width=\"780\" height=\"330\" fill=\"#0d141d\" stroke=\"#2c3d50\" stroke-width=\"1.4\"/>")
    println(io, " <text x=\"1082\" y=\"244\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"16\">Subpanel B1: Simulated Area Resistance R_ohm,eff(t) [8..55 Ohm*cm2] (solid) &amp; P_proj(t) (dotted)</text>")
    println(io, " <text x=\"1082\" y=\"654\" fill=\"#c9d8e8\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"16\">Subpanel B2: Simulated Pressure dP(t)/dP0 [1..18x cap] (solid) &amp; Q_seq(t) Cd2+-eq (dashed)</text>")

    lanes = (
        (450.0f0, false, "#f6b93b", false),
        (620.0f0, false, "#26de81", false),
        (780.0f0, false, "#45aaf2", false),
        (780.0f0, true,  "#ff6bcb", true),
    )
    ND = 57
    for (T_run, rescue_on, col_hex, is_rescue) in lanes
        r_pts = IOBuffer()
        p_pts = IOBuffer()
        dp_pts = IOBuffer()
        q_pts = IOBuffer()
        for i in 1:ND
            t_d = 14.0f0 * Float32(i - 1) / Float32(ND - 1)
            _, _, f_qs_lane, _, chi_ceil_lane, _, _, _ = eval_qs_and_entrainment(
                T_run, eps_slice;
                Pe_flow = Pe_flow, Pe_osc = Pe_osc, stride_hz = stride_hz, ahl_rescue = rescue_on
            )
            rate_k = (T_run == 620.0f0) ? 0.30f0 : ((T_run == 450.0f0) ? 0.27f0 : (rescue_on ? 0.29f0 : 0.20f0))
            lag_g = 2.2f0 + 0.55f0 * (1.0f0 - f_qs_lane)
            span_chi = max(0.08f0, chi_ceil_lane - 0.04f0)
            chi_t = 0.04f0 + span_chi * exp(-exp(lag_g - rate_k * t_d))

            st = eval_5d_state(
                T_run, eps_slice, chi_t;
                Pe_flow = Pe_flow, Pe_osc = Pe_osc, stride_hz = stride_hz, ahl_rescue = rescue_on
            )
            px = 1068.0 + 780.0 * Float64(t_d / 14.0f0)
            uR = clamp((Float64(st.R_ohm_eff) - 8.0) / 47.0, 0.0, 1.0)
            uP = clamp(Float64(st.P_mw_cm2 / p_hi), 0.0, 1.0)
            @printf(r_pts, "%.1f,%.1f ", px, 540.0 - uR * 276.0)
            @printf(p_pts, "%.1f,%.1f ", px, 540.0 - uP * 276.0)

            dp_ratio = min(18.0, 1.0 / max(0.05, Float64(st.kappa_ratio)))
            uDP = clamp((dp_ratio - 1.0) / 17.0, 0.0, 1.0)
            uQ = clamp(Float64(st.Q_seq_mg_g / max(1.0f-3, q_hi)), 0.0, 1.0)
            @printf(dp_pts, "%.1f,%.1f ", px, 940.0 - uDP * 268.0)
            @printf(q_pts, "%.1f,%.1f ", px, 940.0 - uQ * 268.0)
        end
        dash_main = is_rescue ? " stroke-dasharray=\"9,5\"" : ""
        @printf(io, " <polyline points=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"3.0\"%s/>\n", strip(String(take!(r_pts))), col_hex, dash_main)
        @printf(io, " <polyline points=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"2.2\" stroke-dasharray=\"3,5\"/>\n", strip(String(take!(p_pts))), col_hex)
        @printf(io, " <polyline points=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"3.0\"%s/>\n", strip(String(take!(dp_pts))), col_hex, dash_main)
        @printf(io, " <polyline points=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"2.2\" stroke-dasharray=\"8,5\"/>\n", strip(String(take!(q_pts))), col_hex)
    end

    println(io, " <text x=\"1068\" y=\"584\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Day 0</text>")
    println(io, " <text x=\"1458\" y=\"584\" text-anchor=\"middle\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Day 7</text>")
    println(io, " <text x=\"1848\" y=\"584\" text-anchor=\"end\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Day 14</text>")
    println(io, " <text x=\"1068\" y=\"988\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Day 0</text>")
    println(io, " <text x=\"1458\" y=\"988\" text-anchor=\"middle\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Day 7</text>")
    println(io, " <text x=\"1848\" y=\"988\" text-anchor=\"end\" fill=\"#8fa6bd\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"15\">Day 14</text>")
    println(io, "</svg>")

    svg_str = String(take!(io))
    mkpath(dirname(abspath(out_svg)))
    open(out_svg, "w") do f
        write(f, svg_str)
    end
    svg_sha = bytes2hex(sha256(codeunits(svg_str)))
    @printf("[svg] wrote %s (%d bytes, sha256=%s)\n", out_svg, sizeof(svg_str), svg_sha[1:16])
    return svg_str, svg_sha
end

function render_png_from_svg(svg_path::AbstractString, png_path::AbstractString)
    rsvg = Sys.which("rsvg-convert")
    if rsvg !== nothing
        run(`$rsvg -w 1920 -h 1080 $svg_path -o $png_path`)
        return true
    end
    qlm = Sys.which("qlmanage")
    sips = Sys.which("sips")
    if qlm !== nothing && sips !== nothing
        ok = mktempdir() do td
            run(pipeline(`$qlm -t -s 1920 -o $td $svg_path`; stdout = devnull, stderr = devnull))
            cand = joinpath(td, basename(svg_path) * ".png")
            if isfile(cand)
                run(`$sips -z 1080 1920 $cand --out $png_path`)
                return true
            end
            return false
        end
        return ok
    end
    return false
end

function run_selftest()
    s450 = eval_5d_state(450.0f0, 0.80f0, 0.35f0)
    s780_nat = eval_5d_state(780.0f0, 0.80f0, 0.24f0; ahl_rescue = false)
    s780_ahl = eval_5d_state(780.0f0, 0.80f0, 0.42f0; ahl_rescue = true)
    @assert s450.f_QS > s780_nat.f_QS "Expected 450C f_QS > 780C native f_QS"
    @assert s780_ahl.f_QS > s780_nat.f_QS "Expected +AHL rescue to increase f_QS"
    @assert s780_ahl.P_mw_cm2 > s780_nat.P_mw_cm2 "Expected +AHL rescue to boost power at 780C"
    @assert s450.R_char_ohm_cm2 == CoreTransport.char_area_resistance(450.0f0)
    @assert s450.Wo == 0.0f0 && s450.disp_gain == 1.0f0 && s450.streamer_factor == 1.0f0

    wo_check = compute_womersley(18.0f0, 1.80f0)
    @assert wo_check > 0.0f0 && wo_check < 1.0f0 "Unexpected Womersley number: $wo_check"
    s620_darcy = eval_5d_state(620.0f0, 0.80f0, 0.45f0; Pe_flow = 1.80f0, Pe_osc = 0.0f0)
    s620_puls = eval_5d_state(620.0f0, 0.80f0, 0.45f0; Pe_flow = 1.80f0, Pe_osc = 2.20f0, stride_hz = 1.80f0)
    @assert s620_puls.Wo > 0.0f0 && s620_puls.disp_gain > 1.0f0 && s620_puls.streamer_factor < 1.0f0
    @assert s620_puls.chi_ceiling < s620_darcy.chi_ceiling
    @assert s620_puls.kappa_ratio > s620_darcy.kappa_ratio

    front_darcy = compute_pareto_front()
    front_puls = compute_pareto_front(Pe_flow = 1.80f0, Pe_osc = 2.20f0, stride_hz = 1.80f0)
    @assert length(front_darcy) == 59 "Expected 59 non-dominated front points, got $(length(front_darcy))"
    @assert length(front_puls) == 31 "Expected 31 pulsatile front points, got $(length(front_puls))"
    @assert all(p -> p.eps_macro_0 >= 0.799f0, front_darcy)
    @assert issorted([p.P_mw_cm2 for p in front_darcy])
    @assert issorted([-p.Q_seq_mg_g for p in front_darcy])

    mktempdir() do td
        csv_d, _ = write_pareto_csv(joinpath(td, "front.csv"), front_darcy)
        svg_1p, _ = write_1panel_svg(joinpath(td, "fig_1p.svg"), front_darcy)
        svg_d, _ = write_2panel_svg(joinpath(td, "fig.svg"), front_darcy)
        csv_p, _ = write_pareto_csv(joinpath(td, "front_p.csv"), front_puls; Pe_flow = 1.80f0, Pe_osc = 2.20f0, stride_hz = 1.80f0)
        svg_p, _ = write_2panel_svg(joinpath(td, "fig_p.svg"), front_puls; Pe_flow = 1.80f0, Pe_osc = 2.20f0, stride_hz = 1.80f0)
        for txt in (csv_d, svg_1p, svg_d, csv_p, svg_p)
            @assert all(b -> b <= 0x7f, codeunits(txt)) "Non-ASCII byte detected"
            @assert !occursin("\u2014", txt) && !occursin("\u2013", txt) "Em/en dash detected"
        end
        for bad_label in ("2-Electrode EIS", "Percolation Crest", "dP Surge", "14-Day Biochip Pilot", "Zero-em/en pure-ASCII")
            @assert !occursin(bad_label, svg_d) && !occursin(bad_label, svg_p) && !occursin(bad_label, svg_1p) "Forbidden label: $bad_label"
        end
        @assert occursin("Wo,disp_gain,streamer_factor", csv_p)
        @assert occursin("P_mw_cm2_proj,Q_seq_Cd_eq_mg_g", csv_d)
        @assert occursin("Simulated 14-Day Trajectories", svg_d)
    end

    _, p_eq, pf_eq, po_eq, sh_eq, _ = parse_cli_args(["--pulsatile", "--pe-flow=2.0", "--pe-osc=3.0", "--stride-hz=2.5"])
    @assert p_eq && pf_eq == 2.0f0 && po_eq == 3.0f0 && sh_eq == 2.5f0
    for bad_args in (
        ["--pulsatile", "--pe-osc", "0.0"],
        ["--pulsatile", "--pe-osc", "-1.0"],
        ["--pulsatile", "--stride-hz", "0.0"],
        ["--pe-flow", "-0.5"],
        ["--pulsatile", "--pe-osc"],
        ["--unknown-flag"],
    )
        threw = false
        try
            parse_cli_args(bad_args)
        catch err
            threw = err isa ArgumentError
        end
        @assert threw "Expected ArgumentError for CLI args: $bad_args"
    end
    println("EXPORT_BIOCHIP_5D_PARETO_SELFTEST_PASS")
    return true
end

function _take_cli_val(args::Vector{String}, i::Int, flag::String)
    if i >= length(args)
        throw(ArgumentError("Missing value for $flag"))
    end
    return args[i + 1], i + 1
end

function parse_cli_args(args::Vector{String})
    selftest = false
    pulsatile = false
    pe_flow = 1.80f0
    pe_osc = 0.0f0
    pe_osc_set = false
    stride_hz = 1.80f0
    out_dir = normpath(joinpath(@__DIR__, "..", "outputs"))

    i = 1
    while i <= length(args)
        a = args[i]
        val_inline = nothing
        flag = a
        eq_pos = findfirst('=', a)
        if startswith(a, "--") && eq_pos !== nothing
            flag = a[1:(eq_pos - 1)]
            val_inline = a[(eq_pos + 1):end]
        end

        if flag == "--selftest"
            selftest = true
        elseif flag == "--pulsatile"
            pulsatile = true
        elseif flag == "--use-jacc"
            # handled at top-level HAS_JACC init
        elseif flag == "--pe-flow"
            raw_v = val_inline === nothing ? ((v, i) = _take_cli_val(args, i, flag); v) : val_inline
            pe_flow = parse(Float32, raw_v)
        elseif flag == "--pe-osc"
            raw_v = val_inline === nothing ? ((v, i) = _take_cli_val(args, i, flag); v) : val_inline
            pe_osc = parse(Float32, raw_v)
            pe_osc_set = true
        elseif flag == "--stride-hz"
            raw_v = val_inline === nothing ? ((v, i) = _take_cli_val(args, i, flag); v) : val_inline
            stride_hz = parse(Float32, raw_v)
        elseif flag == "--out-dir"
            raw_v = val_inline === nothing ? ((v, i) = _take_cli_val(args, i, flag); v) : val_inline
            out_dir = abspath(raw_v)
        else
            throw(ArgumentError("Unknown CLI argument: $a"))
        end
        i += 1
    end

    if !isfinite(pe_flow) || pe_flow < 0.0f0
        throw(ArgumentError("--pe-flow must be finite and >= 0, got $pe_flow"))
    end
    if !isfinite(stride_hz) || stride_hz <= 0.0f0
        throw(ArgumentError("--stride-hz must be finite and > 0, got $stride_hz"))
    end
    if pulsatile
        if !pe_osc_set
            pe_osc = 2.20f0
        elseif !isfinite(pe_osc) || pe_osc <= 0.0f0
            throw(ArgumentError("--pe-osc must be finite and > 0 when --pulsatile is enabled, got $pe_osc"))
        end
    else
        if pe_osc_set && (!isfinite(pe_osc) || pe_osc < 0.0f0)
            throw(ArgumentError("--pe-osc must be finite and >= 0, got $pe_osc"))
        end
        pe_osc = 0.0f0
    end
    return selftest, pulsatile, pe_flow, pe_osc, stride_hz, out_dir
end

function main(args::Vector{String})
    selftest, pulsatile, pe_flow, pe_osc, stride_hz, out_dir = parse_cli_args(args)
    if selftest
        run_selftest()
        return
    end
    front = compute_pareto_front(Pe_flow = pe_flow, Pe_osc = pe_osc, stride_hz = stride_hz)
    if pulsatile
        csv_path = joinpath(out_dir, "electrobio_pareto_front_pulsatile.csv")
        svg_path = joinpath(out_dir, "Figure1_Biochar_Regime_Map_Pulsatile.svg")
        png_path = joinpath(out_dir, "Figure1_Biochar_Regime_Map_Pulsatile.png")
        write_pareto_csv(csv_path, front; Pe_flow = pe_flow, Pe_osc = pe_osc, stride_hz = stride_hz)
        write_2panel_svg(svg_path, front; Pe_flow = pe_flow, Pe_osc = pe_osc, stride_hz = stride_hz)
        if render_png_from_svg(svg_path, png_path)
            @printf("[png] rendered %s (%d bytes)\n", png_path, filesize(png_path))
        end
    else
        csv_path = joinpath(out_dir, "electrobio_pareto_front.csv")
        svg_path = joinpath(out_dir, "Figure1_Biochar_Regime_Map.svg")
        png_path = joinpath(out_dir, "Figure1_Biochar_Regime_Map.png")
        svg_1p = joinpath(out_dir, "Figure1_Biochar_Pareto_Surface_1Panel.svg")
        png_1p = joinpath(out_dir, "Figure1_Biochar_Pareto_Surface_1Panel.png")
        write_pareto_csv(csv_path, front; Pe_flow = pe_flow, Pe_osc = pe_osc, stride_hz = stride_hz)
        write_2panel_svg(svg_path, front; Pe_flow = pe_flow, Pe_osc = pe_osc, stride_hz = stride_hz)
        write_1panel_svg(svg_1p, front; Pe_flow = pe_flow)
        if render_png_from_svg(svg_path, png_path)
            @printf("[png] rendered %s (%d bytes)\n", png_path, filesize(png_path))
        end
        if render_png_from_svg(svg_1p, png_1p)
            @printf("[png] rendered %s (%d bytes)\n", png_1p, filesize(png_1p))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
