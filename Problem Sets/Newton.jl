using LinearAlgebra
function Newton(f, f_prime; x_0, tolerance=1.0E-10, maxiter=10000)
    x_old = x_0
    normdiff = Inf
    iter = 1
    while normdiff > tolerance && iter <= maxiter
        x_new = x_old - (f(x_old) / f_prime(x_old))
        normdiff = norm(x_new - x_old)
        x_old = x_new
        iter = iter + 1
    end
    return (value=x_old, normdiff=normdiff, iter=iter)
end

f(x) = (x - 1)^3
f_prime(x) = 3 * (x - 1)^2
g(x) = x^2 + 3 * x - 10
g_prime(x) = 2 * x + 3
sol_f = Newton(f, f_prime, x_0 = 0)
sol_g = Newton(g, g_prime, x_0 = 0)
sol_f, sol_g