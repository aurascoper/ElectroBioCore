using Test
using SHA
import ElectroBioCore

# Include exporter in Main without re-including conflicting bindings
include(joinpath(@__DIR__, "..", "scripts", "export_biochip_5d_pareto.jl"))

@testset "ElectroBioCore Suite" begin
    @testset "Core Transport & Unified Char Resistance" begin
        @test ElectroBioCore.CoreTransport.selftest()
        cp = ElectroBioCore.default_params(ElectroBioCore.MODE_BIOCHAR_MFC)
        @test cp.mode == ElectroBioCore.MODE_BIOCHAR_MFC
        cp_con, pb_con = ElectroBioCore.biochar_consolidated_params(680.0f0, 0.68f0, 48.0f0)
        @test 0.0f0 < pb_con.chi < 0.94f0
        @test pb_con.R_char == ElectroBioCore.char_area_resistance(680.0f0)
        @test cp_con.R_ohm > 0.0f0
    end

    @testset "Viscoelastic EPS Rheology & README Benchmark Exactness" begin
        # High-T QS-inhibited monolayer vs +AHL rescue
        r_inh = ElectroBioCore.eval_eps_rheology(0.24, 0.1358; f_drive_Hz = 1.25)
        r_res = ElectroBioCore.eval_eps_rheology(0.42, 0.8400; f_drive_Hz = 1.25)
        @test r_inh.tan_delta > 2.5
        @test r_res.G_storage_Pa > 5.0 * r_inh.G_storage_Pa
        @test r_res.tan_delta < 0.60
        @test r_res.tau_yield_Pa > r_inh.tau_yield_Pa

        front = compute_pareto_front()
        @test length(front) == 59
        p_first = front[1]
        p_mid = front[30]
        p_peak = front[end]
        s780_nat = eval_5d_state(780.0f0, 0.80f0, 0.24f0; ahl_rescue = false)
        s780_res = eval_5d_state(780.0f0, 0.80f0, 0.42f0; ahl_rescue = true)

        expected = (
            (p_first,  400.00, 0.8887, 0.4248, 41.45, 19.68, 0.475, 0.19019, 28.024),
            (p_mid,    488.89, 0.7982, 0.3981, 33.51, 18.59, 0.555, 0.21295, 22.906),
            (p_peak,   584.13, 0.5413, 0.3981, 30.02, 18.49, 0.616, 0.24727, 18.387),
            (s780_nat, 780.00, 0.1358, 0.2400,  4.06, 12.53, 3.086, 0.12021, 10.636),
            (s780_res, 780.00, 0.8400, 0.4200, 39.49, 19.47, 0.493, 0.22240, 15.745),
        )
        for (st, t_exp, fqs_exp, chi_exp, gp_exp, gpp_exp, td_exp, p_exp, q_exp) in expected
            r = ElectroBioCore.eval_eps_rheology(st.chi, st.f_QS; f_drive_Hz = 1.25)
            @test round(Float64(st.T_pyro_C); digits = 2) == t_exp
            @test round(Float64(st.f_QS); digits = 4) == fqs_exp
            @test round(Float64(st.chi); digits = 4) == chi_exp
            @test round(r.G_storage_Pa; digits = 2) == gp_exp
            @test round(r.G_loss_Pa; digits = 2) == gpp_exp
            @test round(r.tan_delta; digits = 3) == td_exp
            @test round(Float64(st.P_mw_cm2); digits = 5) == p_exp
            @test round(Float64(st.Q_seq_mg_g); digits = 3) == q_exp
        end
    end

    @testset "Scripts & Synthetic Recovery Self-Tests" begin
        pkg_root = joinpath(@__DIR__, "..")
        for (rel_path, marker) in (
            ("scripts/export_biochip_5d_pareto.jl", "EXPORT_BIOCHIP_5D_PARETO_SELFTEST_PASS"),
            ("scripts/fit_lignocellulosic_curves.jl", "FIT_LIGNOCELLULOSIC_CURVES_SELFTEST_PASS"),
            ("scripts/sweep_biochar_jacc.jl", "SWEEP_BIOCHAR_JACC_SELFTEST_PASS"),
        )
            script_p = joinpath(pkg_root, rel_path)
            @test isfile(script_p)
            out = read(`$(Base.julia_cmd()) --project=$pkg_root $script_p --selftest`, String)
            @test occursin(marker, out)
        end
    end

    @testset "Byte-Exact Output Regeneration & SHA-256 Determinism" begin
        pkg_root = joinpath(@__DIR__, "..")
        out_dir = joinpath(pkg_root, "outputs")
        mktempdir() do tmpdir
            front_d = compute_pareto_front()
            write_pareto_csv(joinpath(tmpdir, "electrobio_pareto_front.csv"), front_d)
            write_2panel_svg(joinpath(tmpdir, "Figure1_Biochar_Regime_Map.svg"), front_d)
            write_1panel_svg(joinpath(tmpdir, "Figure1_Biochar_Pareto_Surface_1Panel.svg"), front_d)

            front_p = compute_pareto_front(Pe_flow = 1.80f0, Pe_osc = 2.20f0, stride_hz = 1.80f0)
            write_pareto_csv(
                joinpath(tmpdir, "electrobio_pareto_front_pulsatile.csv"), front_p;
                Pe_flow = 1.80f0, Pe_osc = 2.20f0, stride_hz = 1.80f0
            )
            write_2panel_svg(
                joinpath(tmpdir, "Figure1_Biochar_Regime_Map_Pulsatile.svg"), front_p;
                Pe_flow = 1.80f0, Pe_osc = 2.20f0, stride_hz = 1.80f0
            )

            for fname in (
                "electrobio_pareto_front.csv",
                "Figure1_Biochar_Regime_Map.svg",
                "Figure1_Biochar_Pareto_Surface_1Panel.svg",
                "electrobio_pareto_front_pulsatile.csv",
                "Figure1_Biochar_Regime_Map_Pulsatile.svg",
            )
                committed = joinpath(out_dir, fname)
                @test isfile(committed)
                @test read(joinpath(tmpdir, fname)) == read(committed)
            end
        end

        for csv_name in ("electrobio_pareto_front.csv", "electrobio_pareto_front_pulsatile.csv")
            raw = read(joinpath(out_dir, csv_name))
            @test all(b -> b <= 0x7f, raw)
            nl = findfirst(==(UInt8('\n')), raw)
            hdr = String(raw[1:(nl - 1)])
            @test startswith(hdr, "# sha256_body=")
            expected = split(hdr, "=")[2]
            actual = bytes2hex(sha256(raw[(nl + 1):end]))
            @test expected == actual
        end
    end
end
