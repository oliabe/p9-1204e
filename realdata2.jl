using Random, XLSX, LinearAlgebra, Optim, ShiftedArrays, Statistics, Plots, JLD2, DataFrames, Dates
Random.seed!(3)
∑ = x->sum(x)

function W(n, μ, σ, X₀, Δₙ)
    μ, σ, X₀, Δₙ = float([μ, σ, X₀, Δₙ])
    ΔX = μ * Δₙ .+ σ * sqrt(Δₙ) * randn(n)  # Change in log price
    return [X₀, X₀ .+ cumsum(ΔX)...]
end

L = function(w,Σ₋,μ₊,μ;λ₁=1e6,λ₂=1e6)
    return w'*Σ₋*w+λ₁*(∑(w)-1)^2+λ₂*(w'*μ₊-μ)^2
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

RV = (Δx,t) -> Δx[t,:] * Δx[t,:]' #Function for generating elements in the sum in RV

C = (x,t) -> ∑([RV(x,i) for i in 1:t])

c = (kₙ,Δₙ,Δx,j) -> 1/(kₙ*Δₙ)*∑([Δx[i+j,:] * Δx[i+j,:]' for i in 0:kₙ-1])

k = n -> Int(floor(log(n)))

c2 = (kₙ,Δₙ,x,t) -> (C(x[t+kₙ,:]) - C(x[t,:]))/(kₙ*Δₙ)

cont_range = function(returnsdf; tol=1)
    breakpoints = findall(diff(returnsdf.Timestamp) .> Minute(tol)) #Same result as using up to 9 hours
    range = breakpoints |> 
                diff |> 
                argmax |> 
                x->breakpoints[x]+2:breakpoints[x+1]       
    return range
end

wₜ = function(Σ₋,μ₊,μ; trace=false, λ₁=1e6,λ₂=1e6)
    # Initial guess
    n = length(μ₊)
    initial_weights = fill(1/n,n)
    # Portfolio optimization
    # Set bounds
    lower_bounds, upper_bounds = fill(0,n), fill(1,n)

    # Perform the optimization without showing the trace
    opt = Optim.Options(show_trace=trace)
    #L(w,Σ₋,X,μ
    #Σ₋ = C(Matrix(returns[1:t,:]))
    #Σ₋ = cov(Matrix(returns[1:t,:]))
    
    result = optimize(w -> L(w, Σ₋, μ₊, μ; λ₁=λ₁,λ₂=λ₂), 
                      lower_bounds, 
                      upper_bounds, 
                      initial_weights, 
                      Fminbox(LBFGS()), 
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

QV = returns->cumsum(returns.^2,dims=1)
years = function(timestamps)
    m = 365*24*60^2*1e3      
    return broadcast(i->(timestamps[i]- timestamps[1]).value/m, eachindex(timestamps))
end

years = timestamps -> 
        broadcast(i->(timestamps[i]- timestamps[1]).value/(365*24*60^2*1e3), eachindex(timestamps))


QVplot = function(timestamps,returns; base="",date="",year=true, save=false, folder="", show_plot=true, sep=false)
    if !isempty(folder)
        if !isdir("plots/$(folder)")
            mkdir("plots/$(folder)")
        end
        folder = folder*"/"
    end
    t = years(timestamps)
    Δₙ = t[end] / size(returns,1)
    kₙ = k(size(returns,1))
    if !year
        t = timestamps
    end

    quadratic_variation = QV(returns)
    P = plot(legend=:bottomright)
    for i in axes(returns,2)
        if sep || i==1
            P = plot(legend=:bottomright) # Initialize the plot
            if eltype(timestamps)==Time
                xlabel!(P, "Time")
            elseif typeof(timestamps[1])==DateTime
                xlabel!(P,"Date")
            else
                xlabel!(P, "Time in years")
            end
            ylabel!(P, "Quadratic variation")
        end
        # Actual QV
        plot!(P, t[1:size(returns,1)-kₙ], quadratic_variation[1:size(returns,1)-kₙ, i], color = i==1 ? "black" : i, label=(i==1 || sep) ? "Real" : "")

        # Theoretical QV
        #x = [i for i in 1:size(returns,1)-kₙ]
        #plot!(P, t[1:size(returns,1)-kₙ], Δₙ*cumsum([c(kₙ,Δₙ,returns[:,i],j)[1] for j in x]), color = i==1 ? "black" : i, linestyle=:dash, label=(i==1 || sep) ? "Theoretical" : "")
        #plot!(P, t[1:size(returns,1)-kₙ], vol3, color = i==1 ? "black" : i, linestyle=:dash, label=(i==1 || sep) ? "Theoretical" : "")
        
        if sep || i == size(returns,2)
            if save
                savefig(P, "plots/$(folder)$(base)$(date)_QV$(sep ? quotes[i] : "").png")
            end
            if show_plot
                display(P)
            end
        end
    end
end

getquotes = function(base,date; min_n = 1)
    cd("data")
    files = filter(f -> occursin("_$(base)", f) && occursin("_$(date)",f), readdir())
    quotes = broadcast(m->m.captures[1], match.("$(base)([A-Z]{3})"|>Regex, files))
    data = [DataFrame(XLSX.readtable(i, 1, header=false))[:, 1:2]
    for i in files]
    l = [nrow(data[i]) for i in eachindex(files)]

    while nrow(innerjoin(data..., on=:A, makeunique=true)) < min_n
        i = argmin(l)
        deleteat!(data,i); deleteat!(quotes,i); deleteat!(l,i)
    end
    # Merge on timestamps
    prices = innerjoin(data..., on=:A, makeunique=true)
    rename!(prices, ["Timestamp", quotes...])
    cd("..")
    p = Matrix{Float64}(prices[!,2:end])
    
    logprices = log.(p)
    returns = diff(logprices,dims=1)

    returnsdf = returns |> 
    x->hcat(prices[2:end,:Timestamp],x) |>
    x->DataFrame(x, names(prices))

    return quotes,prices, p, logprices, returnsdf, returns
end

quotes, prices, p, logprices, returnsdf, returns = getquotes("EUR","202309", min_n=28000)

r = cont_range(returnsdf, tol=6)

cont_prices = prices[r,:]; cont_p = p[r,:]; cont_logprices = logprices[r,:]; cont_returnsdf = returnsdf[r,:]; cont_returns=returns[r,:]


P1 = select(prices,Not([:JPY]))
p1 = P1[!,2:end] |> Matrix{Float64}
P2 = select(prices,[:Timestamp, :JPY])
p2 = P2[!,2:end] |> Matrix{Float64}


if !isdir("plots/prices")
    mkdir("plots/prices")
end
P = plot(prices.Timestamp, p1, label=permutedims(names(P1)[2:end]),legend=(0.9,0.5))
xlabel!("Date")
ylabel!("Price of €")
savefig(P, "plots/prices/EUR202309_prices1.png")

P = plot(prices.Timestamp, prices.JPY, label="JPY",legend=:bottomleft)
xlabel!("Date")
ylabel!("Price of €")
savefig(P, "plots/prices/EUR202309_prices3.png")





P1 = select(cont_prices,Not([:JPY]))
p1 = P1[!,2:end] |> Matrix{Float64}
P2 = select(cont_prices,[:Timestamp, :JPY])
p2 = P2[!,2:end] |> Matrix{Float64}


if !isdir("plots/cont_prices")
    mkdir("plots/cont_prices")
end
P = plot(cont_prices.Timestamp, p1, label=permutedims(names(P1)[2:end]),legend=(0.9,0.5))
xlabel!("2023-09-22")
ylabel!("Price of €")
savefig(P, "plots/cont_prices/EUR202309_cont_prices1.png")

P = plot(cont_prices.Timestamp, cont_prices.JPY, label="JPY",legend=:bottomleft)
xlabel!("2023-09-22")
ylabel!("Price of €")
savefig(P, "plots/cont_prices/EUR202309_cont_prices3.png")





#no breaks over 6 minutes
QVplot(cont_returnsdf.Timestamp, cont_returns, year = false, save=false, sep=false, base="EUR", date="202309")

#all data
QVplot(returnsdf.Timestamp, returns, save=true, folder="202309",sep=false, year=false, base="EUR", date="202309")

weights = function(μ_year,returns, method="realized")
    μ = (1+μ_year)^(1/(12*size(returns,1)))-1
    allweights = DataFrame(fill(NaN,size(returns,1),size(returns,2)+2),[quotes...,"minimum","converged"])
    if method=="realized"
        
    end
    Σ₋ = zeros(size(returns,2),size(returns,2))
    
    for t in axes(returns,1)
        Σ₋ = Σ₋ .+ RV(returns,t)
        μ₊ = -(returns[t,:] - returns[1,:])
        if t%(floor(size(returns,1)/100)) == 0
            print("\r", round(floor((t / size(returns,1)) * 100),digits=2), "%")
        end
        candidate = wₜ(Σ₋,μ₊,μ)
        w,f,c = Optim.minimizer(candidate), Optim.minimum(candidate), Optim.converged(candidate)
        if !c
            @warn "Candidate did not converge at t=$t."
            #println("Trace:")
            #weights(t,μ; λ₂=1e6, trace=true)
            println("Weights:  $w, Function value: $f")
        end

        allweights[t, :] = [w...,f,c]
    end
    return allweights
end



plot_weights = function(timestamps,weights, quotes; avg=1, alpha=1)
    w = filter(row -> all(x -> !isnan(x), row), weights)[!,Symbol.(quotes)] |> Matrix
    cumw = cumsum(w, dims = 2)
    w_avg = vcat(
            [ mean(cumw[1+i:i+avg,:],dims=1) 
              for i in 0:size(w,1)-avg]...)
    t = timestamps[avg:end]
    # Start plotting with the last column
    p = plot(t,w_avg[:, end], label=quotes[end], fill=(0, :auto), alpha=alpha, yticks=[round(1/(2*size(w,2))*i,digits=2) for i in 0:2*size(w,2)])
    # Plot remaining columns in reverse order
    for i in (size(w_avg, 2)-1):-1:1
        plot!(t,w_avg[:, i], label=quotes[i], fill=(0, :auto), alpha=alpha)
    end
    display(p)
end

allweights = weights(0,cont_returns)

plot_weights(allweights,quotes,avg=1)
savefig("cont_weights.png")
plot_weights(allweights,quotes,avg=60)
savefig("cont_weights_avg60.png")

# Save data
@save "plots/cont_202309/weights_mu=0.1_"*join(quotes, "_")*".jld2" allweights




allweights = weights(0.1,returns)

plot_weights(cont_prices.Timestamp,allweights,quotes,avg=1)
savefig("cont_weights_6min.png")
plot_weights(cont_prices.Timestamp,allweights,quotes,avg=60)
savefig("cont_weights_6min_avg60.png")


# Save data
@save "plots/cont_202309/weights_mu=0.1_"*join(quotes, "_")*".jld2" allweights



















# Create a new column for the day of the week
prices[!, :DayOfWeek] = Dates.dayofweek.(prices.Timestamp)

# Filter out the data for Saturdays and Sundays (6 = Saturday, 7 = Sunday)
cleaned_prices = prices[.!((prices.DayOfWeek .== 6) .| (prices.DayOfWeek .== 7)), :]

# You can then drop the DayOfWeek column if not needed
select!(cleaned_prices, Not(:DayOfWeek))

cleaned_p = Matrix(select(cleaned_prices,Not(:Timestamp)))
quadratic_variation = cumsum(diff(cleaned_p, dims=1).^2,dims=1)

plot(cleaned_prices.Timestamp[2:end],quadratic_variation ./ std.(eachcol(cleaned_p))' * 12)





using DataFrames, Interpolations, Dates

# Sample dataframe (similar to yours)
prices = DataFrame(
    Timestamp = DateTime.(["2023-09-01T00:00:00", "2023-09-01T00:01:00", "2023-09-03T00:00:00"]),
    EURUSD = [1.08416, 1.08414, 1.05719]
)

# Create a full range of timestamps at 1-minute intervals
full_timestamps = DataFrame(Timestamp = collect(DateTime("2023-09-01T00:00:00"):Dates.Minute(1):DateTime("2023-09-03T00:00:00")))

# Left join with the original prices
full_data = leftjoin(full_timestamps, prices, on=:Timestamp)

# Interpolate missing values
for col in names(full_data)[2:end]
    # Create an interpolation object
    itp = LinearInterpolation(dropmissing(full_data, col).Timestamp, dropmissing(full_data, col)[!, col], extrapolation_bc=Line())
    
    # Fill missing values using the interpolation
    full_data[ismissing.(full_data[!, col]), col] = itp.(full_data[ismissing.(full_data[!, col]), :Timestamp])
end

# Display or use the full_data DataFrame as required
