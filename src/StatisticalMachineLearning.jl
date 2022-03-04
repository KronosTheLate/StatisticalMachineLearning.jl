module StatisticalMachineLearning

using Statistics
using Random
import NearestNeighbors: knn
import Base: show

export Picture, TrainTestSplit, remove_constant, flatten, unflatten, visualize_picture, classify, knn, knn_acc, datamat

"""
    Picture(ID::Int, class::Int, data::Vector{T})

A type to hold the information about a picture.
"""
struct Picture{T}
    ID::Int
    class::Int
    data::AbstractVector{T}
    function Picture(x::AbstractVector{<:Real})
        try
            Int64(sqrt(length(x[begin+2:end])))
        catch e
            error("The input vector, with the first two values removed, is not square")
        end
        return new{eltype(x)}(x[1], x[2], x[3:end])
    end
    Picture(a::Int, b::Int, c::Vector{T}) where {T<:Real} = new{T}(a, b, c)
end

"""
    TrainTestSplit(pics::AbstractVector{Picture}, train_to_test_ratio::Rational, shuffle_pics::Bool=true)
    TrainTestSplit(ratio::Rational, n::Int, train::AbstractVector{Picture}, test::AbstractVector{Picture{T}})
"""
struct TrainTestSplit{T}
    ratio::Rational
    n::Int
    train::AbstractVector{Picture{T}}
    test::AbstractVector{Picture{T}}
    TrainTestSplit(ratio::Rational, n::Int, train::AbstractVector{Picture{T}}, test::AbstractVector{Picture{T}}) where {T<:Real} = new{T}(ratio, n, train, test)
    TrainTestSplit(train::AbstractVector{Picture{T}}, test::AbstractVector{Picture{T}}) where {T<:Real} = new{T}(length(train)//length(test), length(train)+length(test), train, test)
    function TrainTestSplit(pics::AbstractVector{Picture{T}}, train_to_test_ratio::Rational, shuffle_pics::Bool=true) where {T<:Real}
    if shuffle_pics
        shuffled_inds = shuffle(pics|>eachindex)
        pics = pics[shuffled_inds]
    end
    n_parts = train_to_test_ratio.num + train_to_test_ratio.den

    trainpics = pics[begin : train_to_test_ratio.num//n_parts * end ÷1]
    testpics = pics[(end - train_to_test_ratio.den//n_parts * end ÷1 +1) : end]
    return new{T}(train_to_test_ratio, length(pics), trainpics, testpics)
    end
end
testclasses(tts::TrainTestSplit) = getfield.(tts.test, :class)
trainclasses(tts::TrainTestSplit) = getfield.(tts.train, :class)

show(io::IO, p::Picture) = println(io, "A $(p.class) drawn by $(p.ID)")
show(io::IO, tts::TrainTestSplit) = println(io, "A TrainTestSplit object with $(tts.n) entries, of train to test ratio $(tts.ratio)")

"""
    flatten(m::AbstractMatrix) = vcat(eachcol(m)...)
"""
flatten(m::AbstractMatrix) = vcat(eachcol(m)...)
function unflatten(x::AbstractArray)
    try
        sidelength = sqrt(length(x)) |> Int64
    catch e
        "The sidelength (`x|>length|>sqrt`) could not be converted to an integer. "
    end
    batches = [x[sidelength*(i-1)+1:sidelength*i] for i in 1:sidelength]
    return hcat(batches...)
end

"""
    visualize_picture(data::AbstractMatrix, colormap::Symbol=:viridis)
    visualize_picture(data::AbstractVector, colormap::Symbol=:viridis)
    visualize_picture(p::Picture, colormap::Symbol=:viridis)
"""
function visualize_picture(data::AbstractMatrix, colormap::Symbol=:viridis)
    fig, ax = heatmap(data, figure=(resolution=(400, 400),); colormap)
    hidedecorations!(ax)
    fig
end
function visualize_picture(data::AbstractVector, colormap::Symbol=:viridis)
    fig, ax = heatmap(unflatten(data)|>reverse|>x->reverse(x, dims=1), figure=(resolution=(400, 400),); colormap)
    hidedecorations!(ax)
    fig
end
function visualize_picture(p::Picture, colormap::Symbol=:viridis)
    fig, ax = heatmap(unflatten(p.data)|>reverse|>x->reverse(x, dims=1), figure=(resolution=(400, 400),),
        axis=(title="Student ID: $(p.ID|>Int64)\nGround truth: $(p.class|>Int64)", ); colormap
    )
    hidedecorations!(ax)
    fig
end

"""
    classify(neighbor_inds::Vector{Int}, train_classes::Vector{Int}; tiebreaker=rand, possible_classes=unique(train_classes))

kwargs:
`tiebreaker` is 
1) a function that takes a tuple of candidates and returns a value, or 
2) a value that is returned upon a tie.
"""
function classify(neighbor_inds::Vector{Int}, train_classes::Vector{Int}, possible_classes=unique(train_classes); tiebreaker=rand)
    neighbor_classes = train_classes[neighbor_inds]
    my_counts = [count(==(possible_classes[i]), neighbor_classes) for i in eachindex(possible_classes)]
    A = [possible_classes my_counts]
    sorted_counts = sortslices(A, dims=1, by=x->x[2], rev=true)
    if sorted_counts[1, 2] == sorted_counts[2, 2]
        inds = [sorted_counts[i, 2] == sorted_counts[1, 2] for i in 1:size(sorted_counts, 1)]
        candidates_of_equal_count = sorted_counts[inds, :][:, 1]
        if tiebreaker isa Function
            return candidates_of_equal_count |> tiebreaker
        else
            f = (args...)->tiebreaker
            return candidates_of_equal_count |> f
        end
    else
        return sorted_counts[1, 1]
    end
end
function classify(neighbor_inds::Vector{Vector{Int}}, args...; kwargs...)
    [classify(neighbor_inds[i], args...; kwargs...) for i in eachindex(neighbor_inds)]
end

function knn(train_pics::Vector{Picture{T}}, test_pics::Vector{Picture{T}}; k::Int, tree=BruteTree, metric=Euclidean(), leafsize::Int=10) where {T<:Real}
    if tree == BruteTree
        mytree = tree(hcat(getfield.(train_pics, :data)...), metric)
    else
        mytree = tree(hcat(getfield.(train_pics, :data)...), metric; leafsize)
    end
    return knn(mytree, hcat(getfield.(test_pics, :data)...), k, true)
end
knn(tts::TrainTestSplit{<:Real}; kwargs...) = knn(tts.train, tts.test; kwargs...)

"""
    knn_acc(tts::TrainTestSplit{<:Real}; tiebreaker=rand, kwargs...)

`kwargs...` are passed on as `k::Int, tree=BruteTree, metric=Euclidean()`
"""
function knn_acc(tts::TrainTestSplit{<:Real}; tiebreaker=rand, kwargs...)
	inds, _ = knn(tts.train, tts.test; kwargs...)
	preds = classify(inds, trainclasses(tts); tiebreaker)
	return mean(preds .== testclasses(tts))
end

"""
	knn_acc_crossvalidate(pics::Vector{Picture}, ratio::Rational = 9//1)

Shuffle `pics` ratio.num + ratio.den times, make a TrainTestSplit for each, and calculate
the mean and standard deviation of the classification accuracy of each.
"""
function knn_acc_crossvalidate(pics::Vector{Picture{T}}, ratio::Rational = 9//1) where {T<:Real}
	n_runs = ratio.num + ratio.den
	temp_TTSs = [TrainTestSplit(pics, ratio) for _ in 1:n_runs]
    x = [knn_acc(tts, k=3) for tts in temp_TTSs]
    (mean=mean(x), std=std(x), n_runs = n_runs)
end

"""
    datamat(pics::Vector{Picture}) = hcat(getfield.(pics, :data)...)

Return the data of all pictures in `pics`.
A column in the returned Matrix represents a single picture.
"""
datamat(pics::Vector{Picture{T}}) where {T<:Real} = hcat(getfield.(pics, :data)...)
datamat(tts::TrainTestSplit{T}) where {T<:Real} = (train=datamat(tts.train), test=datamat(tts.test))

function remove_constant(m::AbstractMatrix)
    bad_row_inds = Int64[]
    all_row_inds = 1:size(m, 1) |> Vector
    for i in all_row_inds
        if maximum(m[i, :]) == minimum(m[i, :])
            push!(bad_row_inds, i)
        end
    end
    reduced_data = m[deleteat!(all_row_inds, bad_row_inds), :]
    return reduced_data
end

function remove_constant(pics::Vector{Picture{T}}) where {T<:Real}
    datatogether = datamat(pics)
    reduced_datamat = remove_constant(datatogether)
    return [Picture(pics[i].ID, pics[i].class, reduced_datamat[:, i]) for i in eachindex(pics)]
end

end
