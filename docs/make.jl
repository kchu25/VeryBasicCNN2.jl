using VeryBasicCNN2
using Documenter

DocMeta.setdocmeta!(VeryBasicCNN2, :DocTestSetup, :(using VeryBasicCNN2); recursive=true)

makedocs(;
    modules=[VeryBasicCNN2],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="VeryBasicCNN2.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/VeryBasicCNN2.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/VeryBasicCNN2.jl",
    devbranch="main",
)
