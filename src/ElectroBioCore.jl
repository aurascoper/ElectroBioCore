# ElectroBioCore.jl - Porous Biochar Electro-Fermentation Transport,
# Quorum-Sensing Signal Quenching, and Viscoelastic Biofilm Rheology.
# Pure ASCII (0 non-ASCII bytes, em=0, en=0).

module ElectroBioCore

include("CoreTransport.jl")
using .CoreTransport:
    MODE_BIOCHAR_MFC,
    MODE_BIO_H2_MEC,
    CoreParams,
    PyrolysisBiocharParams,
    EdgeSensitivities,
    char_area_resistance,
    default_params,
    biochar_consolidated_params,
    edge_weight_sensitivities,
    step_core,
    simulate_trajectory

include("ViscoelasticRheology.jl")
using .ViscoelasticRheology:
    ViscoelasticState,
    eval_eps_rheology

export CoreTransport, ViscoelasticRheology
export MODE_BIOCHAR_MFC, MODE_BIO_H2_MEC
export CoreParams, PyrolysisBiocharParams, EdgeSensitivities
export char_area_resistance, default_params, biochar_consolidated_params, edge_weight_sensitivities
export step_core, simulate_trajectory
export ViscoelasticState, eval_eps_rheology

end # module ElectroBioCore
