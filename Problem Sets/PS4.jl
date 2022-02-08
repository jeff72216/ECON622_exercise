# Exercise 1
using LinearAlgebra, Random
using BenchmarkTools

N = 50
A = rand(N, N)
B = rand(N, N)
@btime $A * $B

# loop by the row first
function rowloop(a, b)
    @assert size(a)[2] == size(b)[1]
    c = zeros(size(a)[1], size(b)[2])
    for j in 1:size(b)[2]
        for i in 1:size(a)[1]
            c[i, j] = a[i, :]' * b[:, j]
        end
    end
    return c
end
@btime rowloop($A, $B)

# loop by the column first
function colloop(a, b)
    @assert size(a)[2] == size(b)[1]
    c = zeros(size(a)[1], size(b)[2])
    for i in 1:size(a)[1]
        for j in 1:size(b)[2]
            c[i, j] = a[i, :]' * b[:, j]
        end
    end
    return c
end
@btime colloop($A, $B)

# loop by the dot product
function rowloopdot(a, b)
    @assert size(a)[2] == size(b)[1]
    c = zeros(size(a)[1], size(b)[2])
    for j in 1:size(b)[2]
        for i in 1:size(a)[1]
            c[i, j] = dot(a[i, :], b[:, j])
        end
    end
    return c
end
@btime rowloopdot($A, $B)

function colloopdot(a, b)
    @assert size(a)[2] == size(b)[1]
    c = zeros(size(a)[1], size(b)[2])
    for i in 1:size(a)[1]
        for j in 1:size(b)[2]
            c[i, j] = dot(a[i, :], b[:, j])
        end
    end
    return c
end
@btime colloopdot($A, $B)

# Compare colloopdot(a, b) to the built-in function
A10 = rand(10, 10)
B10 = rand(10, 10)
A1000 = rand(1000, 1000)
B1000 = rand(1000, 1000)
@btime $A10 * $B10
@btime colloopdot($A10, $B10)
@btime $A1000 * $B1000
@btime colloopdot($A1000, $B1000)