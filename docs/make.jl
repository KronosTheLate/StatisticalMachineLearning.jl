using StatisticalMachineLearning
using Documenter

DocMeta.setdocmeta!(StatisticalMachineLearning, :DocTestSetup, :(using StatisticalMachineLearning); recursive=true)

makedocs(;
    modules=[StatisticalMachineLearning],
    authors="KronosTheLate",
    repo="https://github.com/KronosTheLate/StatisticalMachineLearning.jl/blob/{commit}{path}#{line}",
    sitename="StatisticalMachineLearning.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://KronosTheLate.github.io/StatisticalMachineLearning.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/KronosTheLate/StatisticalMachineLearning.jl",
    devbranch="master",
)
