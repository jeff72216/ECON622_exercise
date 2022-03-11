# This code is modified from part of my master thesis. I tried to use the semiparametric method to 
# develop a new Feasible-GLS estimating procedure. In the first step, a standard FGLS estimating
# procedure is implemented with a non-parametrically identified skedactic function Σ. Then, I resemple
# the test statistic using the wild bootstrap method, and then compare the FGLS t-statistic to the
# artificially-generated distribution. This method is proved to have great performance in finite samples
# in terms of lower size distortion and better testing power. The following code demonstrates the method
# using k-nearest neighbors estimator in the first step. Also, a simple leave-one-out cross validation
# is applied to choose the optimal bandwidth.

using LinearAlgebra, Distributions, StatsBase, BenchmarkTools

function KNN(h, X, y, e)
    #-----------------------------------------------------#
    # h : bandwidth                                       #
    # X : Nx5 matrix of constant and explanatory variable #
    # y : Nx1 matrix of explained variable                #
    # e : Nx1 matrix of residuals                         #
    #-----------------------------------------------------#
    N = length(y)
    w = zeros(N, N)
    for pp in 1:N
        x_normsort = sortperm(view.(Ref(w), 1:size(w,1), :))
        v = x_normsort .<= h
        v[pp] = 0
        w[:, pp] = v ./ sum(v)
    end
    s2hat = transpose(w) * e.^2
    small = mean(e.^2) / 10
    s2hat[s2hat .< small] .= small
    return(s2hat)
end

function CrossValidation(H, X, y, e)
    #-----------------------------------------------------#
    # H : vector of bandwidth candidates                  #
    #-----------------------------------------------------#
    CV = []
    for r in 1:length(H)
        s2hat_cv = KNN(H[r], X, y, e)
        error = sum((e.^2 .- s2hat_cv).^2)
        CV = append!(CV, error)
    end
    h = H[argmin(CV)]
    s2hat = KNN(h, X, y, e)
    Σ = Diagonal(s2hat)
    β̂ = (inv(transpose(X) * inv(Σ) * X) * transpose(X) * inv(Σ) * y)
    return(tuple(β̂, Σ))
end

function WildBootstrap(iter, X, y, e, β̂, Σ)
    N = length(y)
    t_memory = []
    for b in 1:iter
        u = float(rand(Bernoulli(0.5), 100))
        u[u.==0] .= -1
        ỹ = X * β̂ + e.*u
        β̃5 = (inv(transpose(X) * inv(Σ) * X) * transpose(X) * inv(Σ) * ỹ)[5, :]
        tstat = β̃5 / sqrt(inv(transpose(X) * inv(Σ) * X)[5, 5])
        t_memory = append!(t_memory, tstat)
    end
    return(sort(t_memory))
end

# DGP
x1 = ones(100)
x25 = rand(Normal(3, 1), 100, 4)
X = hcat(x1, x25)
β = [1, 1, 1, 1, 0]
ε = rand(100)
y = X * β + ε
β̂ = inv(transpose(X) * X) * transpose(X) * y
e = y - X * β̂

# FGLS estimation
H = range(1, stop = 8, length = 10) * (2.50773668907 + 0.021528881383 * (100^(4 / 5)))
FGLS = CrossValidation(H, X, y, e)
β̂5 = FGLS[1][5, :]
Σ = FGLS[2]
tstat = β̂5 / sqrt(inv(transpose(X) * inv(Σ) * X)[5, 5])

# Resempling t-statistic using wild bootstrap
T = WildBootstrap(1000, X, y, e, β̂, Σ)
uppercrit = percentile(T, 75)
lowercrit = percentile(T, 25)
if tstat[1] >= uppercrit || tstat[1] <= lowercrit
    println("Reject the null hypothesis")
else
    println("Do not reject the null hypothesis")
end

# Here are the executing time of the functions
@btime CrossValidation(H, X, y, e)
@btime WildBootstrap(1000, X, y, e, β̂, Σ)
# The CrossValidation function is slow since it contains multi-layers of loops.

# The following functions use multi-threading to improve the speed.
function KNN_multi(h, X, y, e)
    #-----------------------------------------------------#
    # h : bandwidth                                       #
    # X : Nx5 matrix of constant and explanatory variable #
    # y : Nx1 matrix of explained variable                #
    # e : Nx1 matrix of residuals                         #
    #-----------------------------------------------------#
    N = length(y)
    w = zeros(N, N)
    Threads.@threads for pp in 1:N
        x_normsort = sortperm(view.(Ref(w), 1:size(w,1), :))
        v = x_normsort .<= h
        v[pp] = 0
        w[:, pp] = v ./ sum(v)
    end
    s2hat = transpose(w) * e.^2
    small = mean(e.^2) / 10
    s2hat[s2hat .< small] .= small
    return(s2hat)
end

function CrossValidation_multi(H, X, y, e)
    #-----------------------------------------------------#
    # H : vector of bandwidth candidates                  #
    #-----------------------------------------------------#
    CV = []
    Threads.@threads for r in 1:length(H)
        s2hat_cv = KNN_multi(H[r], X, y, e)
        error = sum((e.^2 .- s2hat_cv).^2)
        CV = append!(CV, error)
    end
    h = H[argmin(CV)]
    s2hat = KNN_multi(h, X, y, e)
    Σ = Diagonal(s2hat)
    β̂ = (inv(transpose(X) * inv(Σ) * X) * transpose(X) * inv(Σ) * y)
    return(tuple(β̂, Σ))
end

# It is around 2 times faster when using 4 threads on my computer
@btime CrossValidation_multi(H, X, y, e)