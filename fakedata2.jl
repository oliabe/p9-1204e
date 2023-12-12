using Random, XLSX, LinearAlgebra, Optim, ShiftedArrays, Statistics, Plots, JLD2, DataFrames
Random.seed!(3)

function W(n, μ, σ, X₀, Δₙ)
    μ, σ, X₀, Δₙ = float([μ, σ, X₀, Δₙ])
    ΔX = μ * Δₙ .+ σ * sqrt(Δₙ) * randn(n)  # Change in log price
    return [X₀, X₀ .+ cumsum(ΔX)...]
end

L = function(w,Σ₋,X,μ;λ₁=1e6,λ₂=1e6)
    μ₊ = X[end,:] - X[1,:]
    return w'*Σ₋*w+λ₁*(sum(w)-1)^2+λ₂*(w'*μ₊-μ)^2
end

function is_positive_definite(A::Matrix) 
    try
        cholesky(A)
        return true
    catch e
        if isa(e, PosDefException)
            return false
        end
        throw(e)  # rethrow if some other exception
    end
end

weights = function(t,μ; trace=false, λ₁=1e6,λ₂=1e6)
    #μ = 0.001
    # Initial guess
    initial_weights = fill(1/ncol(returns),ncol(returns))
    # Portfolio optimization
    # Set bounds
    lower_bounds, upper_bounds = fill(0,ncol(returns)), fill(1,ncol(returns))

    # Perform the optimization without showing the trace
    opt = Optim.Options(show_trace=trace)
    #L(w,Σ₋,X,μ
    Σ₋ = cov(Matrix(returns[1:t,:]))
    X = Matrix(returns[1:t,:])
    result = optimize(w -> L(w, Σ₋, X, μ; λ₁=λ₁,λ₂=λ₂), 
                      lower_bounds, 
                      upper_bounds, 
                      initial_weights, 
                      Fminbox(BFGS()), 
                      opt)

    #result = optimize(objective, constraint1, constraint2, initial_weights, Fminbox(); lower = [bounds[i][1] for i=1:2], upper = [bounds[i][2] for i=1:2])
    return result
end

targetsearch = function(μ0; max_restarts=200)
    μ = μ0
    allweights = copy(returns)
    allweights[!,:converged] .= 0
    allweights[!,:minimum] .= 0

    counter = 0
    t = 3
    while counter < max_restarts
        while t<=400
            if t%(floor(nrow(returns)/100)) == 0
                print("\r", round(floor((t / nrow(returns)) * 100),digits=2), "%")
            end
            candidate = weights(t,μ)
            if !Optim.converged(candidate)
                break
            end
            t+=1
        end
        if t > 100  # If inner loop went up to and including 10
            break
        else
            μ /= 2
            counter += 1
            t = 3
        end
    end
    return μ
end


# Parameters
n = Int(30 * 24 * 60) # Assuming 30 trading days and 24 hours of trading per day  
μ = 0.1  # Annual drift rate
σ = 1  # Annual volatility 
T = 30/365  # Annual time horizon
X₀ = 10 #Initial price
Δₙ = T/n #Discretization coefficient
d = 5   # Number of quotes

logprices = DataFrame(hcat([W(n, μ, σ, X₀, Δₙ) for _ in 1:d]...), :auto)

plot(range(0,30,n+1),Matrix(logprices))
savefig("plots/fake_logprices")

quotes = names(logprices)

returns = DataFrame(zeros(n,d),quotes)


for q in quotes
    returns[!, Symbol(q)] = diff(logprices[!,Symbol(q)])
end

quadratic_variation = cumsum(Matrix(returns).^2,dims=1)
plot(range(0,30,n),quadratic_variation/T, legend=:none) #t
xlabel!("Time in days")
ylabel!("Quadratic variation")
#savefig("plots/fake_qv.png")


#Statistical analysis
using Statistics

for col in names(data)
    mean_col = mean(data[!, col])
    std_col = std(data[!, col])
    println("Column: $col, Mean: $mean_col, Standard Deviation: $std_col")
end

using ARCHModels

# Function to fit GARCH(1,1) model and return estimated volatility
function estimate_volatility(column_data)
    model = TGARCH{1,0,1}
    fit_model = fit(model, column_data)
    return sqrt.(fit_model.spec.coefs)
end

# Applying the function to each column and storing the estimated volatilities
volatilities = DataFrame()
for col in names(data)
    volatilities[!, col] = estimate_volatility(data[!, col])
end

println(first(volatilities, 5))  # Print the first 5 rows of estimated volatilities
plot(Matrix(volatilities), title = "Estimated Volatilities", label = names(volatilities))


function autocorrelation(data, lag)
    mean_data = mean(data)
    n = length(data)
    num = sum((data[1:(n-lag)] .- mean_data) .* (data[(lag+1):n] .- mean_data))
    den = sum((data .- mean_data).^2)
    
    return num / den
end

for col in names(data)
    auto_corr = autocorrelation(data[!, col], 1)
    println("Autocorrelation at lag 1 for $col: $auto_corr")
end

lags = 1:20  # You can adjust the number of lags as needed
autocorrs = [[autocorrelation(data[!, col], lag) for col in names(data)] for lag in lags]


autocorr_matrix = Matrix{Float64}([autocorrelation(data[!, col], lag) for lag in lags, col in names(data)])

function moving_average(data, window_size)
    n = length(data)
    moving_avg = zeros(n)

    for i in window_size:n
        moving_avg[i] = mean(data[i-window_size+1:i])
    end

    return moving_avg
end

window_size = 60  # Example window size

for col in names(data)
    data[!, string(col, "_filtered")] = moving_average(data[!, col], window_size)
end



realized_covariation = function(X)
    ΔⁿX = diff(X,dims=1)
    return cumsum(ΔⁿX*ΔⁿX')
end


μ_year = 0.1
μ = (1+μ_year)^(1/(12*nrow(returns)))-1

allweights = DataFrame(fill(NaN,nrow(returns),ncol(returns)+2),[quotes...,"minimum","converged"])

r = 3:nrow(returns)
for t in r
    if t%(floor(nrow(returns)/100)) == 0
        print("\r", round(floor((t / nrow(returns)) * 100),digits=2), "%")
    end
    candidate = weights(t,μ; λ₂=1e6)
    w,f,c = Optim.minimizer(candidate), Optim.minimum(candidate), Optim.converged(candidate)
    if !c
        @warn "Candidate did not converge at t=$t."
        #println("Trace:")
        #weights(t,μ; λ₂=1e6, trace=true)
        println("Weights:  $w, Function value: $f")
    end

    allweights[t, :] = [w...,f,c]
end

plot_weights = function(weights, quotes; avg=1, alpha=1, n_x=30)
    w = filter(row -> all(x -> !isnan(x), row), weights)[!,Symbol.(quotes)] |> Matrix
    cumw = cumsum(w, dims = 2)
    w_avg = vcat(
            [ mean(cumw[1+i:i+avg,:],dims=1) 
              for i in 0:size(w,1)-avg]...)
    # Start plotting with the last column
    p = plot(range(0,n_x,size(w_avg,1)),w_avg[:, end], label=quotes[end], fill=(0, :auto), alpha=alpha, yticks=[1/(2*size(w,2))*i for i in 0:2*size(w,2)])
    # Plot remaining columns in reverse order
    for i in (size(w_avg, 2)-1):-1:1
        plot!(range(0,30,size(w_avg,1)),w_avg[:, i], label=quotes[i], fill=(0, :auto), alpha=alpha)
    end
    display(p)
end

plot_weights(allweights,quotes, avg=1)
xlabel!("Time in days")
ylabel!("Proportion of asset")

savefig("plots/fake_weights_1.png")


mean(Matrix(allweights),dims=1)
minimum(Matrix(allweights),dims=1)
maximum(Matrix(allweights),dims=1)


# Save data

@save date*"_weights_mu="*string(μ)*"_"*join(quotes, "_")*".jld2" allweights

# Load data
μ = 0.001
weight1 = @load date*"_weights_mu="*string(μ)*"_"*join(quotes, "_")*".jld2" allweights


