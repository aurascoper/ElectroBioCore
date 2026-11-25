#!/usr/bin/env julia
# fit_lignocellulosic_curves.jl
# Pure-ASCII Modified Gompertz & Monod-Electro ODE Curve Fitter for Biochar/Hydrolysate Data.
# Emits deterministic ElectroBioCore.BiocharCalibrationReceipt.v1 JSON receipts.
# Zero external dependencies (Julia Base + LinearAlgebra + Printf + SHA + Statistics).

using LinearAlgebra
using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "..", "src", "CoreTransport.jl"))
using .CoreTransport

# --------------------------------------------------------------------------
# 1. Data Containers
# --------------------------------------------------------------------------
struct MicrobialSeries
    t_h::Vector{Float64}
    substrate::Vector{Float64}
    product::Vector{Float64}
    has_current::Bool
    j_obs::Vector{Float64}
end

struct GompertzFit
    P_max::Float64
    R_max::Float64
    lambda_h::Float64
    se_P_max::Float64
    se_R_max::Float64
    se_lambda_h::Float64
    r2::Float64
    rmse::Float64
end

struct MonodElectroFit
    j_scale::Float32
    K_half::Float32
    k_para::Float32
    se_j_scale::Float32
    se_K_half::Float32
    se_k_para::Float32
    r2_sub::Float64
    rmse_sub::Float64
end

# --------------------------------------------------------------------------
# 2. Pure-Base CSV Loader
# Expected columns: time_h, substrate, product[, current_mA_cm2]
# --------------------------------------------------------------------------
function load_microbial_csv(path::AbstractString)::MicrobialSeries
    t_h = Float64[]
    sub = Float64[]
    prd = Float64[]
    cur = Float64[]
    has_cur = false
    open(path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue
            startswith(line, "#") && continue
            parts = split(line, ',')
            length(parts) < 3 && continue
            t_val = tryparse(Float64, strip(parts[1]))
            t_val === nothing && continue
            s_val = parse(Float64, strip(parts[2]))
            p_val = parse(Float64, strip(parts[3]))
            push!(t_h, t_val)
            push!(sub, s_val)
            push!(prd, p_val)
            if length(parts) >= 4
                j_val = tryparse(Float64, strip(parts[4]))
                if j_val !== nothing
                    has_cur = true
                    push!(cur, j_val)
                end
            end
        end
    end
    n = length(t_h)
    @assert n >= 5 "CSV must contain at least 5 numeric rows (got $n)"
    return MicrobialSeries(t_h, sub, prd, has_cur && length(cur) == n, has_cur ? cur : Float64[])
end

# --------------------------------------------------------------------------
# 3. Damped Levenberg-Marquardt Optimizer (Box-Projected + Jacobian SEs)
# --------------------------------------------------------------------------
function lm_fit(
    resid_fn::Function,
    theta0::Vector{Float64},
    lb::Vector{Float64},
    ub::Vector{Float64};
    max_iter::Int = 160,
    tol::Float64 = 1.0e-9
)
    p = length(theta0)
    theta = clamp.(copy(theta0), lb, ub)
    r = resid_fn(theta)
    m = length(r)
    cost = 0.5 * dot(r, r)
    lambda = 1.0e-3
    J = zeros(Float64, m, p)

    for _ in 1:max_iter
        for j in 1:p
            h = max(1.0e-6, 1.0e-5 * abs(theta[j]))
            tp = copy(theta); tp[j] = min(ub[j], theta[j] + h)
            tm = copy(theta); tm[j] = max(lb[j], theta[j] - h)
            denom = max(1.0e-12, tp[j] - tm[j])
            J[:, j] = (resid_fn(tp) - resid_fn(tm)) ./ denom
        end
        g = J' * r
        if norm(g, Inf) < tol
            break
        end
        H = J' * J
        diag_scale = [max(1.0e-6, H[j, j]) for j in 1:p]
        A = H + lambda * Diagonal(diag_scale)
        step = -(A \ g)
        cand = clamp.(theta .+ step, lb, ub)
        r_cand = resid_fn(cand)
        cost_cand = 0.5 * dot(r_cand, r_cand)
        if cost_cand < cost
            rel_imp = (cost - cost_cand) / max(cost, 1.0e-12)
            theta = cand
            r = r_cand
            cost = cost_cand
            lambda = max(1.0e-8, 0.35 * lambda)
            if rel_imp < tol && norm(step) < 1.0e-7 * (1.0 + norm(theta))
                break
            end
        else
            lambda = min(1.0e8, 4.0 * lambda)
        end
    end

    dof = max(1, m - p)
    s2 = (2.0 * cost) / Float64(dof)
    H_fin = J' * J + 1.0e-10 * Matrix{Float64}(I, p, p)
    cov_mat = s2 .* inv(H_fin)
    se = [sqrt(max(0.0, cov_mat[j, j])) for j in 1:p]
    return theta, se, r
end

function r2_and_rmse(y_obs::Vector{Float64}, resid::Vector{Float64})
    ss_res = dot(resid, resid)
    y_mean = mean(y_obs)
    ss_tot = sum((y - y_mean)^2 for y in y_obs)
    r2 = ss_tot > 1.0e-15 ? (1.0 - ss_res / ss_tot) : 1.0
    rmse = sqrt(ss_res / Float64(length(y_obs)))
    return r2, rmse
end

# --------------------------------------------------------------------------
# 4. Modified Gompertz Curve Fit
# --------------------------------------------------------------------------
@inline function gompertz_eval(t::Float64, P_max::Float64, R_max::Float64, lambda_h::Float64)
    u = (R_max * MathConstants.e / max(P_max, 1.0e-9)) * (lambda_h - t) + 1.0
    u_clamped = clamp(u, -40.0, 3.5)
    return P_max * exp(-exp(u_clamped))
end

function fit_modified_gompertz(ds::MicrobialSeries)::GompertzFit
    p_span = max(1.0e-4, maximum(ds.product))
    t_span = max(1.0e-3, ds.t_h[end] - ds.t_h[1])
    max_slope = p_span / t_span
    lag_guess = 0.08 * t_span
    for i in 2:length(ds.t_h)
        dt = max(1.0e-6, ds.t_h[i] - ds.t_h[i - 1])
        sl = (ds.product[i] - ds.product[i - 1]) / dt
        if sl > max_slope
            max_slope = sl
            lag_guess = clamp(ds.t_h[i - 1] - ds.product[i - 1] / max(sl, 1.0e-6), 0.0, 0.7 * t_span)
        end
    end

    theta0 = [1.05 * p_span, max_slope, lag_guess]
    lb = [0.25 * p_span, 1.0e-5 * (p_span / t_span), 0.0]
    ub = [4.00 * p_span, 20.0 * max_slope, 0.85 * ds.t_h[end]]

    resid_fn = function(th)
        P_m, R_m, lam = th[1], th[2], th[3]
        return [gompertz_eval(ds.t_h[i], P_m, R_m, lam) - ds.product[i] for i in eachindex(ds.t_h)]
    end

    th, se, r = lm_fit(resid_fn, theta0, lb, ub)
    r2, rmse = r2_and_rmse(ds.product, r)
    return GompertzFit(th[1], th[2], th[3], se[1], se[2], se[3], r2, rmse)
end

# --------------------------------------------------------------------------
# 5. Monod + Parasitic Sink ODE Fit -> ElectroBioCore (j_scale, K_half, k_para)
# --------------------------------------------------------------------------
function integrate_monod_series(
    t_h::Vector{Float64},
    S0::Float64,
    P0::Float64,
    j_scale::Float64,
    K_half::Float64,
    k_para::Float64,
    lambda_h::Float64,
    conv_sub_per_h::Float64,
    conv_prd_per_h::Float64
)
    n = length(t_h)
    S_pred = Vector{Float64}(undef, n)
    P_pred = Vector{Float64}(undef, n)
    J_pred = Vector{Float64}(undef, n)
    S = S0
    P = P0
    S_pred[1] = S
    P_pred[1] = P
    act0 = 1.0 / (1.0 + exp(-2.5 * (t_h[1] - lambda_h)))
    J_pred[1] = j_scale * act0 * (max(0.0, S) / (K_half + max(0.0, S) + 1.0e-9))

    for i in 2:n
        dt_tot = max(0.0, t_h[i] - t_h[i - 1])
        n_sub = max(4, Int(ceil(dt_tot / 0.20)))
        dt = dt_tot / Float64(n_sub)
        t_cur = t_h[i - 1]
        j_last = J_pred[i - 1]
        for _ in 1:n_sub
            t_mid = t_cur + 0.5 * dt
            f_lag = 1.0 / (1.0 + exp(-2.5 * (t_mid - lambda_h)))
            s_pos = max(0.0, S)
            mon1 = s_pos / (K_half + s_pos + 1.0e-9)
            jt1 = j_scale * f_lag * mon1
            jp1 = k_para * (0.35 + 0.65 * f_lag) * mon1
            dS1 = -conv_sub_per_h * (jt1 + jp1)

            s_mid = max(0.0, S + 0.5 * dt * dS1)
            mon2 = s_mid / (K_half + s_mid + 1.0e-9)
            jt2 = j_scale * f_lag * mon2
            jp2 = k_para * (0.35 + 0.65 * f_lag) * mon2

            S = max(0.0, S - dt * conv_sub_per_h * (jt2 + jp2))
            P = P + dt * conv_prd_per_h * jt2
            j_last = jt2
            t_cur += dt
        end
        S_pred[i] = S
        P_pred[i] = P
        J_pred[i] = j_last
    end
    return S_pred, P_pred, J_pred
end

function fit_monod_electro(
    ds::MicrobialSeries,
    gf::GompertzFit;
    mode::Int32 = MODE_BIOCHAR_MFC,
    prod_per_sub_yield::Float64 = 1.0
)::MonodElectroFit
    base = default_params(mode)
    s_span = max(1.0e-4, ds.substrate[1] - minimum(ds.substrate))
    p_span = max(1.0e-4, maximum(ds.product) - ds.product[1])
    stoich_yield = (p_span / s_span) > 0.02 ? (p_span / s_span) : prod_per_sub_yield

    raw_conv_h = 3600.0 * Float64(base.v_ratio) / Float64(base.n_e)
    t_span_h = max(1.0, ds.t_h[end] - ds.t_h[1])
    time_norm = clamp(4.0 / t_span_h, 0.04, 1.0)
    conv_sub_h = raw_conv_h * time_norm * max(ds.substrate[1], 0.1)
    conv_prd_h = conv_sub_h * stoich_yield

    j0_guess = clamp(gf.R_max / max(conv_prd_h, 1.0e-6), 0.15, 4.50)
    K0_guess = clamp(0.20 * ds.substrate[1], 0.01, 2.50)
    kp_guess = clamp(0.10 * j0_guess, 0.005, 1.20)
    lag0_guess = clamp(gf.lambda_h * 0.80, 0.0, 0.85 * ds.t_h[end])
    yps0_guess = clamp(stoich_yield * 1.08, 0.20, 2.50)

    theta0 = [j0_guess, K0_guess, kp_guess, lag0_guess, yps0_guess]
    lb = [0.05, 0.002 * max(ds.substrate[1], 0.1), 0.001, 0.0, 0.10]
    ub = [8.00, 3.00 * max(ds.substrate[1], 0.1), 2.50, 0.85 * ds.t_h[end], 4.00]

    w_sub = 1.0 / max(ds.substrate[1], 1.0e-4)
    w_prd = 1.0 / max(maximum(ds.product), 1.0e-4)
    w_cur = ds.has_current ? (1.5 / max(maximum(ds.j_obs), 1.0e-4)) : 0.0

    resid_fn = function(th)
        S_p, P_p, J_p = integrate_monod_series(
            ds.t_h, ds.substrate[1], ds.product[1],
            th[1], th[2], th[3], th[4], conv_sub_h, conv_sub_h * th[5]
        )
        n = length(S_p)
        res = Vector{Float64}(undef, ds.has_current ? 3 * n : 2 * n)
        for i in 1:n
            res[i] = w_sub * (S_p[i] - ds.substrate[i])
            res[n + i] = w_prd * (P_p[i] - ds.product[i])
            if ds.has_current
                res[2 * n + i] = w_cur * (J_p[i] - ds.j_obs[i])
            end
        end
        return res
    end

    th, se, _ = lm_fit(resid_fn, theta0, lb, ub)
    S_fin, _, _ = integrate_monod_series(
        ds.t_h, ds.substrate[1], ds.product[1],
        th[1], th[2], th[3], th[4], conv_sub_h, conv_sub_h * th[5]
    )
    sub_resid = S_fin .- ds.substrate
    r2_s, rmse_s = r2_and_rmse(ds.substrate, sub_resid)
    return MonodElectroFit(
        Float32(th[1]), Float32(th[2]), Float32(th[3]),
        Float32(se[1]), Float32(se[2]), Float32(se[3]),
        r2_s, rmse_s
    )
end

# --------------------------------------------------------------------------
# 6. Map Fitted Microbial Kinetics + Biochar State -> CoreParams & Edge Weights
# --------------------------------------------------------------------------
function build_calibrated_core_params(
    mf::MonodElectroFit;
    T_pyro_C::Float32 = 680.0f0,
    eps_0::Float32 = 0.68f0,
    L_f_um::Float32 = 48.0f0,
    E_app::Float32 = -0.06f0,
    mode::Int32 = MODE_BIOCHAR_MFC
)
    cp_cal, pb = biochar_consolidated_params(
        T_pyro_C, eps_0, L_f_um;
        mode = mode,
        base_j_scale = mf.j_scale,
        base_K_half = mf.K_half,
        base_k_para = mf.k_para
    )
    es = edge_weight_sensitivities(
        T_pyro_C, eps_0, L_f_um, E_app;
        base_j_scale = mf.j_scale,
        base_K_half = mf.K_half,
        base_k_para = mf.k_para
    )
    return cp_cal, pb, es
end

# --------------------------------------------------------------------------
# 7. Deterministic Pure-ASCII JSON Receipt Serialization
# --------------------------------------------------------------------------
function json_escape_ascii(s::AbstractString)::String
    io = IOBuffer()
    for ch in s
        if ch == '"'
            write(io, "\\\"")
        elseif ch == '\\'
            write(io, "\\\\")
        elseif UInt32(ch) < 0x20 || UInt32(ch) > 0x7e
            write(io, '?')
        else
            write(io, ch)
        end
    end
    return String(take!(io))
end

@inline json_f6(v::Real) = @sprintf("%.6f", Float64(v))
@inline json_e6(v::Real) = @sprintf("%.6e", Float64(v))

function format_calibration_json(
    csv_source_name::AbstractString,
    csv_sha256::AbstractString,
    n_points::Int,
    has_current::Bool,
    is_synthetic_self_recovery::Bool,
    gf::GompertzFit,
    mf::MonodElectroFit,
    cp::CoreParams,
    pb::PyrolysisBiocharParams,
    es::EdgeSensitivities,
    E_app::Float32
)::String
    c_end, th_end, eta_end, q_end, jt, etaF, p_mw, y_frac, c_err = simulate_trajectory(cp, E_app)
    note_str = is_synthetic_self_recovery ?
        "Synthetic parameter-recovery verification only; not fitted to experimental biochar or hydrolysate data." :
        "Empirical CSV calibration receipt."
    io = IOBuffer()
    println(io, "{")
    println(io, "  \"schema\": \"ElectroBioCore.BiocharCalibrationReceipt.v1\",")
    println(io, "  \"input\": {")
    println(io, "    \"source_name\": \"$(json_escape_ascii(csv_source_name))\",")
    println(io, "    \"csv_sha256\": \"$(json_escape_ascii(csv_sha256))\",")
    println(io, "    \"n_points\": $(n_points),")
    println(io, "    \"has_current_column\": $(has_current ? "true" : "false"),")
    println(io, "    \"is_synthetic_self_recovery\": $(is_synthetic_self_recovery ? "true" : "false"),")
    println(io, "    \"note\": \"$(json_escape_ascii(note_str))\"")
    println(io, "  },")
    println(io, "  \"gompertz_fit\": {")
    println(io, "    \"P_max\": $(json_f6(gf.P_max)),")
    println(io, "    \"R_max_per_h\": $(json_f6(gf.R_max)),")
    println(io, "    \"lambda_lag_h\": $(json_f6(gf.lambda_h)),")
    println(io, "    \"ci95_P_max\": $(json_f6(1.96 * gf.se_P_max)),")
    println(io, "    \"ci95_R_max_per_h\": $(json_f6(1.96 * gf.se_R_max)),")
    println(io, "    \"ci95_lambda_lag_h\": $(json_f6(1.96 * gf.se_lambda_h)),")
    println(io, "    \"r2\": $(json_f6(gf.r2)),")
    println(io, "    \"rmse\": $(json_e6(gf.rmse))")
    println(io, "  },")
    println(io, "  \"monod_electro_fit\": {")
    println(io, "    \"j_scale_mA_cm2\": $(json_f6(mf.j_scale)),")
    println(io, "    \"K_half\": $(json_f6(mf.K_half)),")
    println(io, "    \"k_para_mA_cm2\": $(json_f6(mf.k_para)),")
    println(io, "    \"ci95_j_scale\": $(json_f6(1.96f0 * mf.se_j_scale)),")
    println(io, "    \"ci95_K_half\": $(json_f6(1.96f0 * mf.se_K_half)),")
    println(io, "    \"ci95_k_para\": $(json_f6(1.96f0 * mf.se_k_para)),")
    println(io, "    \"r2_substrate\": $(json_f6(mf.r2_sub)),")
    println(io, "    \"rmse_substrate\": $(json_e6(mf.rmse_sub))")
    println(io, "  },")
    println(io, "  \"pyrolysis_biochar_state\": {")
    println(io, "    \"T_pyro_C\": $(json_f6(pb.T_pyro_C)),")
    println(io, "    \"eps_0\": $(json_f6(pb.eps_0)),")
    println(io, "    \"L_f_um\": $(json_f6(pb.L_f_um)),")
    println(io, "    \"chi\": $(json_f6(pb.chi)),")
    println(io, "    \"eps_eff\": $(json_f6(pb.eps_eff)),")
    println(io, "    \"R_char_ohm_cm2\": $(json_f6(pb.R_char)),")
    println(io, "    \"R_ohm_eff_ohm_cm2\": $(json_f6(pb.R_ohm_eff)),")
    println(io, "    \"K_half_eff\": $(json_f6(pb.K_half_eff)),")
    println(io, "    \"j_max_eff_mA_cm2\": $(json_f6(pb.j_max_eff)),")
    println(io, "    \"k_para_eff_mA_cm2\": $(json_f6(pb.k_para_eff))")
    println(io, "  },")
    println(io, "  \"core_params\": {")
    println(io, "    \"mode\": $(cp.mode),")
    println(io, "    \"E0_V\": $(json_f6(cp.E0_V)),")
    println(io, "    \"j_scale\": $(json_f6(cp.j_scale)),")
    println(io, "    \"alpha\": $(json_f6(cp.alpha)),")
    println(io, "    \"K_half\": $(json_f6(cp.K_half)),")
    println(io, "    \"R_ohm\": $(json_f6(cp.R_ohm)),")
    println(io, "    \"k_para\": $(json_f6(cp.k_para)),")
    println(io, "    \"n_e\": $(json_f6(cp.n_e)),")
    println(io, "    \"tau_rc\": $(json_f6(cp.tau_rc)),")
    println(io, "    \"v_ratio\": $(json_f6(cp.v_ratio)),")
    println(io, "    \"theta_cap\": $(json_f6(cp.theta_cap))")
    println(io, "  },")
    println(io, "  \"edge_sensitivities\": {")
    println(io, "    \"dRohm_dTpyro_ohm_cm2_per_C\": $(json_f6(es.dRohm_dTpyro)),")
    println(io, "    \"dKhalf_dchi\": $(json_f6(es.dKhalf_dchi)),")
    println(io, "    \"dP_dLf_mW_per_cm2_um\": $(json_f6(es.dP_dLf)),")
    println(io, "    \"detaF_dchi\": $(json_f6(es.detaF_dchi))")
    println(io, "  },")
    println(io, "  \"trajectory_verification\": {")
    println(io, "    \"E_app_V\": $(json_f6(E_app)),")
    println(io, "    \"c_bulk_final\": $(json_f6(c_end)),")
    println(io, "    \"theta_final\": $(json_f6(th_end)),")
    println(io, "    \"eta_final_V\": $(json_f6(eta_end)),")
    println(io, "    \"q_cum_mC_cm2\": $(json_f6(q_end)),")
    println(io, "    \"mean_j_target_mA_cm2\": $(json_f6(jt)),")
    println(io, "    \"mean_eta_F\": $(json_f6(etaF)),")
    println(io, "    \"mean_power_mW_cm2\": $(json_f6(p_mw)),")
    println(io, "    \"yield_fraction\": $(json_f6(y_frac)),")
    println(io, "    \"C_phys_mass_error\": $(json_e6(c_err))")
    println(io, "  }")
    println(io, "}")
    return String(take!(io))
end

function emit_calibration_json(
    out_json_path::AbstractString,
    csv_path::AbstractString,
    ds::MicrobialSeries,
    gf::GompertzFit,
    mf::MonodElectroFit,
    cp::CoreParams,
    pb::PyrolysisBiocharParams,
    es::EdgeSensitivities,
    E_app::Float32;
    source_label::Union{Nothing, AbstractString} = nothing,
    is_synthetic_self_recovery::Bool = false
)
    csv_sha = bytes2hex(open(sha256, csv_path))
    src_name = source_label === nothing ? basename(csv_path) : String(source_label)
    payload = format_calibration_json(
        src_name, csv_sha, length(ds.t_h), ds.has_current, is_synthetic_self_recovery,
        gf, mf, cp, pb, es, E_app
    )
    mkpath(dirname(abspath(out_json_path)))
    open(out_json_path, "w") do io
        write(io, payload)
    end
    receipt_sha = bytes2hex(sha256(codeunits(payload)))
    @printf("[json] wrote %s (%d bytes, sha256=%s)\n", out_json_path, sizeof(payload), receipt_sha)
    return payload, receipt_sha
end

function print_calibration_report(
    gf::GompertzFit,
    mf::MonodElectroFit,
    cp::CoreParams,
    pb::PyrolysisBiocharParams,
    es::EdgeSensitivities,
    E_app::Float32
)
    _, _, _, _, jt, etaF, p_mw, y_frac, c_err = simulate_trajectory(cp, E_app)
    println("=== Lignocellulosic Microbial & Biochar Calibration Report ===")
    @printf(
        "Gompertz Fit : P_max=%.4f (+/-%.4f) | R_max=%.4f/h (+/-%.4f) | lag=%.2f h (+/-%.2f) | R2=%.5f | RMSE=%.4e\n",
        gf.P_max, 1.96 * gf.se_P_max, gf.R_max, 1.96 * gf.se_R_max,
        gf.lambda_h, 1.96 * gf.se_lambda_h, gf.r2, gf.rmse
    )
    @printf(
        "Monod Base Fit : j_scale=%.4f (+/-%.4f) mA/cm2 | K_half=%.4f (+/-%.4f) | k_para=%.4f (+/-%.4f) | R2=%.5f\n",
        mf.j_scale, 1.96f0 * mf.se_j_scale, mf.K_half, 1.96f0 * mf.se_K_half,
        mf.k_para, 1.96f0 * mf.se_k_para, mf.r2_sub
    )
    @printf(
        "Biochar State : T_pyro=%.1f C | eps_0=%.3f | L_f=%.1f um | chi=%.3f | eps_eff=%.3f | R_ohm=%.2f Ohm*cm2\n",
        pb.T_pyro_C, pb.eps_0, pb.L_f_um, pb.chi, pb.eps_eff, cp.R_ohm
    )
    @printf(
        "Core Trajectory: E_app=%.3f V | j_target=%.4f mA/cm2 | P=%.4f mW/cm2 | eta_F=%.4f | yield=%.4f | C_phys=%.2e\n",
        E_app, jt, p_mw, etaF, y_frac, c_err
    )
    @printf(
        "Edge Weights : dR_ohm/dT_pyro=%.4f | dK_half/dchi=%.4f | dP/dL_f=%.5f | deta_F/dchi=%.4f\n",
        es.dRohm_dTpyro, es.dKhalf_dchi, es.dP_dLf, es.detaF_dchi
    )
end

function write_synthetic_benchmark_csv(csv_path::AbstractString)
    true_j = 0.94
    true_K = 0.16
    true_kp = 0.075
    true_lag = 3.20
    t_grid = collect(0.0:1.5:36.0)
    base = default_params(MODE_BIOCHAR_MFC)
    raw_conv_h = 3600.0 * Float64(base.v_ratio) / Float64(base.n_e)
    conv_sub_h = raw_conv_h * (4.0 / 36.0) * 1.0
    conv_prd_h = conv_sub_h * 0.88
    S_syn, P_syn, J_syn = integrate_monod_series(
        t_grid, 1.0, 0.0, true_j, true_K, true_kp, true_lag, conv_sub_h, conv_prd_h
    )
    mkpath(dirname(abspath(csv_path)))
    open(csv_path, "w") do io
        println(io, "# time_h,substrate,product,current_mA_cm2")
        for i in eachindex(t_grid)
            wiggle = 0.0015 * sin(0.7 * i)
            s_i = max(0.0, S_syn[i] + wiggle)
            p_i = max(0.0, P_syn[i] - 0.8 * wiggle)
            j_i = max(0.0, J_syn[i] + 0.5 * wiggle)
            @printf(io, "%.3f,%.6f,%.6f,%.6f\n", t_grid[i], s_i, p_i, j_i)
        end
    end
    return true_j, true_K, true_kp
end

# --------------------------------------------------------------------------
# 8. Deterministic Self-Test (--selftest)
# --------------------------------------------------------------------------
function run_selftest()
    mktempdir() do tmpdir
        csv_path = joinpath(tmpdir, "synthetic_lignocellulosic_run.csv")
        json_path1 = joinpath(tmpdir, "receipt_a.json")
        json_path2 = joinpath(tmpdir, "receipt_b.json")
        true_j, true_K, _ = write_synthetic_benchmark_csv(csv_path)

        ds = load_microbial_csv(csv_path)
        gf = fit_modified_gompertz(ds)
        mf = fit_monod_electro(ds, gf; mode = MODE_BIOCHAR_MFC)
        cp, pb, es = build_calibrated_core_params(
            mf; T_pyro_C = 680.0f0, eps_0 = 0.68f0, L_f_um = 48.0f0, E_app = -0.06f0
        )
        _, _, _, _, _, etaF, _, _, c_err = simulate_trajectory(cp, -0.06f0)
        @assert gf.r2 > 0.985 "Gompertz R2 too low: $(gf.r2)"
        @assert mf.r2_sub > 0.990 "Monod substrate R2 too low: $(mf.r2_sub)"
        @assert abs(mf.j_scale - Float32(true_j)) < 0.12f0 "j_scale recovery drift: $(mf.j_scale)"
        @assert abs(mf.K_half - Float32(true_K)) < 0.06f0 "K_half recovery drift: $(mf.K_half)"
        @assert c_err < 1.0f-4 "Calibrated CoreParams violated C_phys: $c_err"
        @assert 0.50f0 < etaF <= 1.0f0 "Invalid coulombic efficiency: $etaF"
        @assert es.dRohm_dTpyro < 0.0f0 && es.dKhalf_dchi > 0.0f0 "Invalid EdgeSensitivities signs"

        payload1, sha1 = emit_calibration_json(
            json_path1, csv_path, ds, gf, mf, cp, pb, es, -0.06f0;
            is_synthetic_self_recovery = true
        )
        payload2, sha2 = emit_calibration_json(
            json_path2, csv_path, ds, gf, mf, cp, pb, es, -0.06f0;
            is_synthetic_self_recovery = true
        )
        @assert sha1 == sha2 && payload1 == payload2 "JSON receipt emission is not deterministic"
        @assert all(b -> b <= 0x7f, codeunits(payload1)) "JSON receipt contains non-ASCII bytes"
        @assert occursin("\"ElectroBioCore.BiocharCalibrationReceipt.v1\"", payload1)
        @assert occursin("\"is_synthetic_self_recovery\": true", payload1)
        @assert occursin("\"gompertz_fit\":", payload1)
        @assert occursin("\"monod_electro_fit\":", payload1)
        @assert occursin("\"core_params\":", payload1)
        @assert occursin("\"edge_sensitivities\":", payload1)
        @assert !occursin("NaN", payload1) && !occursin("Inf", payload1)
        print_calibration_report(gf, mf, cp, pb, es, -0.06f0)
    end
    println("FIT_LIGNOCELLULOSIC_CURVES_SELFTEST_PASS")
    return true
end

# --------------------------------------------------------------------------
# 9. CLI Entrypoint
# --------------------------------------------------------------------------
function main(args::Vector{String})
    if "--selftest" in args
        run_selftest()
        return
    end
    if isempty(args) || ("-h" in args) || ("--help" in args)
        println("Usage:")
        println("  julia fit_lignocellulosic_curves.jl --selftest")
        println("  julia fit_lignocellulosic_curves.jl [<data.csv> | --synthetic-recovery] [--T-pyro 680] [--eps0 0.68] [--Lf 48] [--E-app -0.06] [--emit-json <path>]")
        return
    end

    csv_path = ""
    json_path = ""
    use_syn = false
    T_pyro = 680.0f0
    eps_0 = 0.68f0
    L_f = 48.0f0
    E_app = -0.06f0

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--synthetic-recovery" || a == "--synthetic-benchmark"
            use_syn = true; i += 1
        elseif a == "--T-pyro" && i < length(args)
            T_pyro = parse(Float32, args[i + 1]); i += 2
        elseif startswith(a, "--T-pyro=")
            T_pyro = parse(Float32, split(a, '=', limit = 2)[2]); i += 1
        elseif a == "--eps0" && i < length(args)
            eps_0 = parse(Float32, args[i + 1]); i += 2
        elseif startswith(a, "--eps0=")
            eps_0 = parse(Float32, split(a, '=', limit = 2)[2]); i += 1
        elseif a == "--Lf" && i < length(args)
            L_f = parse(Float32, args[i + 1]); i += 2
        elseif startswith(a, "--Lf=")
            L_f = parse(Float32, split(a, '=', limit = 2)[2]); i += 1
        elseif a == "--E-app" && i < length(args)
            E_app = parse(Float32, args[i + 1]); i += 2
        elseif startswith(a, "--E-app=")
            E_app = parse(Float32, split(a, '=', limit = 2)[2]); i += 1
        elseif a == "--emit-json" && i < length(args)
            json_path = args[i + 1]; i += 2
        elseif startswith(a, "--emit-json=")
            json_path = split(a, '=', limit = 2)[2]; i += 1
        elseif !startswith(a, "--") && isempty(csv_path)
            csv_path = a; i += 1
        else
            error("Unknown or incomplete argument: $a")
        end
    end

    if use_syn
        mktempdir() do tmpdir
            syn_csv = joinpath(tmpdir, "synthetic_self_recovery_input.csv")
            write_synthetic_benchmark_csv(syn_csv)
            ds = load_microbial_csv(syn_csv)
            gf = fit_modified_gompertz(ds)
            mf = fit_monod_electro(ds, gf; mode = MODE_BIOCHAR_MFC)
            cp, pb, es = build_calibrated_core_params(
                mf; T_pyro_C = T_pyro, eps_0 = eps_0, L_f_um = L_f, E_app = E_app
            )
            print_calibration_report(gf, mf, cp, pb, es, E_app)
            if !isempty(json_path)
                emit_calibration_json(
                    json_path, syn_csv, ds, gf, mf, cp, pb, es, E_app;
                    source_label = "synthetic_self_recovery_input.csv",
                    is_synthetic_self_recovery = true
                )
            end
        end
        return
    end

    @assert !isempty(csv_path) "Missing CSV file path (or pass --synthetic-recovery)"
    ds = load_microbial_csv(csv_path)
    gf = fit_modified_gompertz(ds)
    mf = fit_monod_electro(ds, gf; mode = MODE_BIOCHAR_MFC)
    cp, pb, es = build_calibrated_core_params(
        mf; T_pyro_C = T_pyro, eps_0 = eps_0, L_f_um = L_f, E_app = E_app
    )
    print_calibration_report(gf, mf, cp, pb, es, E_app)
    if !isempty(json_path)
        emit_calibration_json(json_path, csv_path, ds, gf, mf, cp, pb, es, E_app)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
