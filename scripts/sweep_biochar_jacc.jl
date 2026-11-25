# sweep_biochar_jacc.jl
# Pure-ASCII CPU / optional JACC.jl parameter sweep for ElectroBioCore.
# Supports --calibration-json <path> (BiocharCalibrationReceipt.v1), --out-lut <path>, --use-jacc, --selftest.

using Printf
using SHA

const HAS_JACC = ("--use-jacc" in ARGS)
if HAS_JACC
    @eval using JACC
    @eval JACC.@init_backend
end

include(joinpath(@__DIR__, "..", "src", "CoreTransport.jl"))
using .CoreTransport

# --------------------------------------------------------------------------
# 0. Zero-Dependency Calibration Receipt Loader (--calibration-json <path>)
# --------------------------------------------------------------------------
struct CalibrationOverride
    active::Bool
    source_path::String
    sha256_short::String
    j_scale::Float32
    K_half::Float32
    k_para::Float32
end

function default_calibration_override()::CalibrationOverride
    b = default_params(MODE_BIOCHAR_MFC)
    return CalibrationOverride(false, "default", "none", b.j_scale, b.K_half, b.k_para)
end

function _extract_json_block(raw::AbstractString, block_name::AbstractString)::SubString{String}
    s = String(raw)
    pat = Regex("\"" * block_name * "\"\\s*:\\s*\\{([^\\}]*)\\}")
    m = match(pat, s)
    m === nothing && return SubString(s, 1, 0)
    return m.captures[1]
end

function _extract_json_float(text::AbstractString, key::AbstractString)::Union{Nothing, Float32}
    isempty(text) && return nothing
    pat = Regex("\"" * key * "\"\\s*:\\s*([-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?)")
    m = match(pat, text)
    m === nothing && return nothing
    v = tryparse(Float32, m.captures[1])
    return (v !== nothing && isfinite(v)) ? v : nothing
end

function load_calibration_json(path::AbstractString)::CalibrationOverride
    isfile(path) || error("Calibration JSON file not found: $path")
    raw_bytes = read(path)
    sha_hex = bytes2hex(sha256(raw_bytes))
    sha_short = sha_hex[1:12]
    raw = String(raw_bytes)

    monod_blk = _extract_json_block(raw, "monod_electro_fit")
    j_val = _extract_json_float(monod_blk, "j_scale_mA_cm2")
    K_val = _extract_json_float(monod_blk, "K_half")
    p_val = _extract_json_float(monod_blk, "k_para_mA_cm2")

    if j_val === nothing || K_val === nothing || p_val === nothing
        cp_blk = _extract_json_block(raw, "core_params")
        j_val = j_val === nothing ? _extract_json_float(cp_blk, "j_scale") : j_val
        K_val = K_val === nothing ? _extract_json_float(cp_blk, "K_half") : K_val
        p_val = p_val === nothing ? _extract_json_float(cp_blk, "k_para") : p_val
    end

    (j_val !== nothing && j_val > 0.0f0) || error("Missing or non-positive j_scale in $path")
    (K_val !== nothing && K_val > 0.0f0) || error("Missing or non-positive K_half in $path")
    (p_val !== nothing && p_val > 0.0f0) || error("Missing or non-positive k_para in $path")

    @printf(
        "[calibration] loaded %s (sha256=%s): j_scale=%.4f mA/cm2 | K_half=%.4f | k_para=%.4f mA/cm2\n",
        basename(path), sha_short, j_val, K_val, p_val
    )
    return CalibrationOverride(true, String(path), sha_short, j_val, K_val, p_val)
end

# --------------------------------------------------------------------------
# 1. Kernels: Coarse 4D + Fine 2D Ridge Sweep for (T_pyro, eps_0, L_opt)
# --------------------------------------------------------------------------
function _biochar_coarse_kernel(
    idx, c_pow, c_etaF, c_cerr,
    n_t::Int32, n_p::Int32, n_l::Int32, n_e::Int32,
    cal_j::Float32, cal_K::Float32, cal_p::Float32
)
    rem0 = idx - Int32(1)
    i_e = mod(rem0, n_e); rem1 = div(rem0, n_e)
    i_l = mod(rem1, n_l); rem2 = div(rem1, n_l)
    i_p = mod(rem2, n_p)
    i_t = div(rem2, n_p)

    T_pyro = 400.0f0 + 450.0f0 * (Float32(i_t) / Float32(n_t - Int32(1)))
    eps_0  = 0.38f0  + 0.42f0  * (Float32(i_p) / Float32(n_p - Int32(1)))
    L_f_um = 8.0f0   + 91.0f0  * (Float32(i_l) / Float32(n_l - Int32(1)))
    e_app  = -0.20f0 + 0.24f0  * (Float32(i_e) / Float32(n_e - Int32(1)))

    cp, _ = biochar_consolidated_params(
        T_pyro, eps_0, L_f_um;
        mode = MODE_BIOCHAR_MFC,
        base_j_scale = cal_j, base_K_half = cal_K, base_k_para = cal_p
    )
    _, _, _, _, _, etaF, p_mw, _, c_err = simulate_trajectory(cp, e_app)
    @inbounds c_pow[idx] = p_mw
    @inbounds c_etaF[idx] = etaF
    @inbounds c_cerr[idx] = c_err
    return nothing
end

function _biochar_fine_kernel(
    idx, f_pow, f_etaF, f_chi, f_rohm, f_cerr,
    f_t::Int32, f_l::Int32, eps_opt::Float32, e_opt::Float32,
    cal_j::Float32, cal_K::Float32, cal_p::Float32
)
    i_t = div(idx - Int32(1), f_l)
    i_l = mod(idx - Int32(1), f_l)
    T_pyro = 450.0f0 + 370.0f0 * (Float32(i_t) / Float32(f_t - Int32(1)))
    L_f_um = 10.0f0  + 90.0f0  * (Float32(i_l) / Float32(f_l - Int32(1)))

    cp, pb = biochar_consolidated_params(
        T_pyro, eps_opt, L_f_um;
        mode = MODE_BIOCHAR_MFC,
        base_j_scale = cal_j, base_K_half = cal_K, base_k_para = cal_p
    )
    _, _, _, _, _, etaF, p_mw, _, c_err = simulate_trajectory(cp, e_opt)
    @inbounds f_pow[idx] = p_mw
    @inbounds f_etaF[idx] = etaF
    @inbounds f_chi[idx] = pb.chi
    @inbounds f_rohm[idx] = pb.R_ohm_eff
    @inbounds f_cerr[idx] = c_err
    return nothing
end

function run_biochar_consolidation_sweep(cal::CalibrationOverride = default_calibration_override())
    cal_j = cal.j_scale
    cal_K = cal.K_half
    cal_p = cal.k_para

    # Pass 1: 4D global sweep (16 T_pyro x 8 eps_0 x 8 L_f x 4 E_app = 4,096)
    N_T = 16; N_P = 8; N_L = 8; N_E = 4
    N_TOT = N_T * N_P * N_L * N_E
    hc_pow = Vector{Float32}(undef, N_TOT)
    hc_etaF = Vector{Float32}(undef, N_TOT)
    hc_cerr = Vector{Float32}(undef, N_TOT)

    if HAS_JACC
        c_pow = JACC.zeros(Float32, N_TOT)
        c_etaF = JACC.zeros(Float32, N_TOT)
        c_cerr = JACC.zeros(Float32, N_TOT)
        JACC.parallel_for(
            N_TOT, _biochar_coarse_kernel,
            c_pow, c_etaF, c_cerr,
            Int32(N_T), Int32(N_P), Int32(N_L), Int32(N_E),
            cal_j, cal_K, cal_p
        )
        hc_pow .= Array(c_pow)
        hc_etaF .= Array(c_etaF)
        hc_cerr .= Array(c_cerr)
    else
        for idx in 1:N_TOT
            _biochar_coarse_kernel(
                Int32(idx), hc_pow, hc_etaF, hc_cerr,
                Int32(N_T), Int32(N_P), Int32(N_L), Int32(N_E),
                cal_j, cal_K, cal_p
            )
        end
    end

    best_c = 1
    best_cpow = -1.0f0
    for idx in 1:N_TOT
        if hc_etaF[idx] >= 0.75f0 && hc_cerr[idx] < 1.0f-4 && hc_pow[idx] > best_cpow
            best_cpow = hc_pow[idx]
            best_c = idx
        end
    end
    rem0 = best_c - 1
    i_e_c = mod(rem0, N_E); rem1 = div(rem0, N_E)
    rem2 = div(rem1, N_L)
    i_p_c = mod(rem2, N_P)
    eps_opt = Float32(0.38f0 + 0.42f0 * (Float32(i_p_c) / Float32(N_P - 1)))
    e_opt   = Float32(-0.20f0 + 0.24f0 * (Float32(i_e_c) / Float32(N_E - 1)))

    # Pass 2: Fine 64 x 64 (T_pyro x L_f) ridge zoom = 4,096
    F_T = 64
    F_L = 64
    F_TOT = F_T * F_L
    hf_pow = Vector{Float32}(undef, F_TOT)
    hf_etaF = Vector{Float32}(undef, F_TOT)
    hf_chi = Vector{Float32}(undef, F_TOT)
    hf_rohm = Vector{Float32}(undef, F_TOT)
    hf_cerr = Vector{Float32}(undef, F_TOT)

    if HAS_JACC
        f_pow = JACC.zeros(Float32, F_TOT)
        f_etaF = JACC.zeros(Float32, F_TOT)
        f_chi = JACC.zeros(Float32, F_TOT)
        f_rohm = JACC.zeros(Float32, F_TOT)
        f_cerr = JACC.zeros(Float32, F_TOT)
        JACC.parallel_for(
            F_TOT, _biochar_fine_kernel,
            f_pow, f_etaF, f_chi, f_rohm, f_cerr,
            Int32(F_T), Int32(F_L), eps_opt, e_opt,
            cal_j, cal_K, cal_p
        )
        hf_pow .= Array(f_pow)
        hf_etaF .= Array(f_etaF)
        hf_chi .= Array(f_chi)
        hf_rohm .= Array(f_rohm)
        hf_cerr .= Array(f_cerr)
    else
        for idx in 1:F_TOT
            _biochar_fine_kernel(
                Int32(idx), hf_pow, hf_etaF, hf_chi, hf_rohm, hf_cerr,
                Int32(F_T), Int32(F_L), eps_opt, e_opt,
                cal_j, cal_K, cal_p
            )
        end
    end

    best_f = 1
    best_fpow = -1.0f0
    max_cerr = max(maximum(hc_cerr), maximum(hf_cerr))
    for idx in 1:F_TOT
        if hf_etaF[idx] >= 0.75f0 && hf_cerr[idx] < 1.0f-4 && hf_pow[idx] > best_fpow
            best_fpow = hf_pow[idx]
            best_f = idx
        end
    end

    it_opt = div(best_f - 1, F_L)
    il_opt = mod(best_f - 1, F_L)
    T_opt = 450.0f0 + 370.0f0 * (Float32(it_opt) / Float32(F_T - 1))
    L_opt = 10.0f0 + 90.0f0 * (Float32(il_opt) / Float32(F_L - 1))
    chi_opt = hf_chi[best_f]
    rohm_opt = hf_rohm[best_f]

    thresh_95 = 0.95f0 * best_fpow
    T_win_lo = 9999.0f0; T_win_hi = -9999.0f0
    L_win_lo = 9999.0f0; L_win_hi = -9999.0f0
    chi_win_hi = 0.0f0
    for idx in 1:F_TOT
        if hf_etaF[idx] >= 0.75f0 && hf_pow[idx] >= thresh_95
            it = div(idx - 1, F_L)
            il = mod(idx - 1, F_L)
            Tv = 450.0f0 + 370.0f0 * (Float32(it) / Float32(F_T - 1))
            Lv = 10.0f0 + 90.0f0 * (Float32(il) / Float32(F_L - 1))
            T_win_lo = min(T_win_lo, Tv); T_win_hi = max(T_win_hi, Tv)
            L_win_lo = min(L_win_lo, Lv); L_win_hi = max(L_win_hi, Lv)
            chi_win_hi = max(chi_win_hi, hf_chi[idx])
        end
    end

    es = edge_weight_sensitivities(
        T_opt, eps_opt, L_opt, e_opt;
        base_j_scale = cal_j, base_K_half = cal_K, base_k_para = cal_p
    )
    @printf(
        "[BIO-MFC optimum] T_pyro*=%.1f C | eps_0*=%.3f | L_opt*=%.1f um | chi*=%.3f | R_ohm*=%.2f Ohm*cm2 | E_app*=%.3f V | P*=%.4f mW/cm2 | eta_F=%.4f | max_C_phys=%.2e\n",
        T_opt, eps_opt, L_opt, chi_opt, rohm_opt, e_opt, best_fpow, hf_etaF[best_f], max_cerr
    )
    @printf(
        "[BIO-MFC 95%% window] T_pyro in [%.1f, %.1f] C | L_f in [%.1f, %.1f] um | chi <= %.3f | eps_0*=%.3f\n",
        T_win_lo, T_win_hi, L_win_lo, L_win_hi, chi_win_hi, eps_opt
    )
    @printf(
        "[Edge sensitivities] dR_ohm/dT_pyro=%.4f | dK_half/dchi=%.4f | dP/dL_f=%.5f | deta_F/dchi=%.4f\n",
        es.dRohm_dTpyro, es.dKhalf_dchi, es.dP_dLf, es.detaF_dchi
    )

    return T_opt, eps_opt, L_opt, chi_opt, rohm_opt, es
end

# --------------------------------------------------------------------------
# 2. Biochar 16-Row LUT Writer
# --------------------------------------------------------------------------
function format_tibasic_func(fn_name::String, rows::Vector{NTuple{4, Float32}})::String
    io = IOBuffer()
    println(io, "Define $(fn_name)(idx)=")
    println(io, "Func")
    println(io, " If idx<1 Then")
    println(io, "  Return $(length(rows))")
    println(io, " EndIf")
    for (k, r) in enumerate(rows)
        cond = (k == 1) ? "If" : "ElseIf"
        @printf(io, " %s idx=%d Then\n", cond, k)
        @printf(io, "  Return {%.4f,%.4f,%.4f,%.4f}\n", r[1], r[2], r[3], r[4])
    end
    println(io, " EndIf")
    r_last = rows[end]
    @printf(io, " Return {%.4f,%.4f,%.4f,%.4f}\n", r_last[1], r_last[2], r_last[3], r_last[4])
    println(io, "EndFunc")
    return String(take!(io))
end

function write_tibasic_bundle(
    out_path::String,
    T_opt::Float32,
    eps_opt::Float32,
    L_opt::Float32,
    chi_opt::Float32,
    rohm_opt::Float32,
    cal::CalibrationOverride = default_calibration_override()
)
    N_LUT = 16
    mfc_rows = Vector{NTuple{4, Float32}}(undef, N_LUT)
    cp_mfc, _ = biochar_consolidated_params(
        T_opt, eps_opt, L_opt;
        mode = MODE_BIOCHAR_MFC,
        base_j_scale = cal.j_scale,
        base_K_half = cal.K_half,
        base_k_para = cal.k_para
    )
    for k in 1:N_LUT
        e_app = -0.24f0 + 0.30f0 * (Float32(k - 1) / Float32(N_LUT - 1))
        _, _, _, _, jt, etaF, p_mw, _, _ = simulate_trajectory(cp_mfc, e_app)
        mfc_rows[k] = (e_app, jt, etaF, p_mw)
    end

    header = @sprintf(
        "# ElectroBioCore Biochar LUT | MFC(T_pyro=%.1fC,eps0=%.2f,Lopt=%.1fum,chi=%.3f,Rohm=%.2f,cal=%s)\n",
        T_opt, eps_opt, L_opt, chi_opt, rohm_opt, cal.sha256_short
    )
    bundle = header * "\n" * format_tibasic_func("eb_lut_mfc", mfc_rows)

    open(out_path, "w") do io
        write(io, bundle)
    end
    lut_sha = bytes2hex(sha256(codeunits(bundle)))
    @printf(
        "[lut] wrote %s (%d bytes, sha256=%s, cal=%s)\n",
        out_path, sizeof(bundle), lut_sha[1:16], cal.sha256_short
    )
    return bundle
end

# --------------------------------------------------------------------------
# 3. Self-Test for Receipt Ingestion (--selftest)
# --------------------------------------------------------------------------
function run_selftest()
    CoreTransport.selftest()
    mktempdir() do tmpdir
        mock_json = joinpath(tmpdir, "mock_calibration_receipt.json")
        mock_lut = joinpath(tmpdir, "mock_bundle.tibasic")
        open(mock_json, "w") do io
            write(io, """
{
  "schema": "ElectroBioCore.BiocharCalibrationReceipt.v1",
  "monod_electro_fit": {
    "j_scale_mA_cm2": 1.020000,
    "K_half": 0.135000,
    "k_para_mA_cm2": 0.068000
  }
}
""")
        end
        cal = load_calibration_json(mock_json)
        @assert cal.active
        @assert isapprox(cal.j_scale, 1.02f0; atol = 1.0f-5)
        @assert isapprox(cal.K_half, 0.135f0; atol = 1.0f-5)
        @assert isapprox(cal.k_para, 0.068f0; atol = 1.0f-5)
        T_opt, eps_opt, L_opt, chi_opt, rohm_opt, _ = run_biochar_consolidation_sweep(cal)
        bundle = write_tibasic_bundle(mock_lut, T_opt, eps_opt, L_opt, chi_opt, rohm_opt, cal)
        @assert occursin("cal=$(cal.sha256_short)", bundle)
        @assert !occursin("eb_lut_hg", bundle)
        @assert all(b -> b <= 0x7f, codeunits(bundle))
    end
    println("SWEEP_BIOCHAR_JACC_SELFTEST_PASS")
    return true
end

# --------------------------------------------------------------------------
# 4. CLI Entrypoint
# --------------------------------------------------------------------------
function main(args::Vector{String})
    if "--selftest" in args
        run_selftest()
        return
    end
    cal_path = ""
    out_lut = normpath(joinpath(@__DIR__, "..", "eb_lut_bundle.tibasic"))
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--use-jacc"
            i += 1
        elseif a == "--calibration-json" && i < length(args)
            cal_path = args[i + 1]; i += 2
        elseif startswith(a, "--calibration-json=")
            cal_path = split(a, '=', limit = 2)[2]; i += 1
        elseif a == "--out-lut" && i < length(args)
            out_lut = args[i + 1]; i += 2
        elseif startswith(a, "--out-lut=")
            out_lut = split(a, '=', limit = 2)[2]; i += 1
        else
            error("Unknown argument: $a")
        end
    end

    CoreTransport.selftest()
    cal = isempty(cal_path) ? default_calibration_override() : load_calibration_json(cal_path)
    T_opt, eps_opt, L_opt, chi_opt, rohm_opt, _ = run_biochar_consolidation_sweep(cal)
    write_tibasic_bundle(out_lut, T_opt, eps_opt, L_opt, chi_opt, rohm_opt, cal)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
