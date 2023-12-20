using Random, XLSX, LinearAlgebra, Optim, ShiftedArrays, Statistics, Plots, JLD2, DataFrames, Dates, Printf
∑ = x->sum(x)

L = function(w,Σ₋,μ₊,μ;λ₁=1e6,λ₂=1e6)
    L = w'*Σ₋*w+λ₁*(∑(w)-1)^2+λ₂*(w'*μ₊-μ)^2
    if !isfinite(L)
        println("not finite:")
        println(w,Σ₋,μ₊,μ)
    end
    return L
end

C = Δx -> cumsum([Δx[t,:] * Δx[t,:]' for t in axes(Δx,1)])

k = n -> Int(ceil(sqrt(n)))

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
        plot!(P, t[1:size(returns,1)-kₙ], quadratic_variation[1:size(returns,1)-kₙ, i], color = i==1 ? "black" : i, label=quotes[i])

        # Theoretical QV
        x = [i for i in 1:size(returns,1)-kₙ]
        plot!(P, t[1:size(returns,1)-kₙ], 
        x*std(returns[1:size(returns,1)-kₙ,i])^2, 
        color = i==1 ? "black" : i, 
        linestyle=:dash, label="")
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

getquotes = function(base,date; min_n = 1, invert = false)
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
    if invert
        prices[!,quotes] = prices[!,quotes].^(-1)
    end
    p = Matrix{Float64}(prices[!,2:end])
    
    logprices = log.(p)
    returns = diff(logprices,dims=1)

    returnsdf = returns |> 
    x->hcat(prices[2:end,:Timestamp],x) |>
    x->DataFrame(x, names(prices))

    return quotes,prices, p, logprices, returnsdf, returns
end

quotes, prices, p, logprices, returnsdf, returns = getquotes("EUR","202309", min_n=28000, invert=false)

r = cont_range(returnsdf, tol=6)

cont_prices6 = prices[r,:]; cont_p6 = p[r,:]; cont_logprices6 = logprices[r,:]; cont_returnsdf6 = returnsdf[r,:]; cont_returns6=returns[r,:]

r = cont_range(returnsdf, tol=1)

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
ylabel!("Price in EUR")
savefig(P, "plots/prices/EUR202309_prices1.png")

P = plot(prices.Timestamp, 100 .* prices.JPY, label="JPY",legend=:none)
xlabel!("Date")
ylabel!("Price of 100 JPY in EUR")
savefig(P, "plots/prices/EUR202309_prices3.png")

P1 = select(cont_prices6,Not([:JPY]))
p1 = P1[!,2:end] |> Matrix{Float64}
P2 = select(cont_prices6,[:Timestamp, :JPY])
p2 = P2[!,2:end] |> Matrix{Float64}

if !isdir("plots/cont_prices")
    mkdir("plots/cont_prices")
end
P = plot(cont_prices6.Timestamp, p1, label=permutedims(names(P1)[2:end]),legend=(0.9,0.5))
xlabel!("Date")
ylabel!("Price in EUR")
savefig(P, "plots/cont_prices/EUR202309_cont6_prices1.png")

P = plot(cont_prices6.Timestamp, 100 .* cont_prices6.JPY, label="JPY",legend=:none)
xlabel!("Date")
ylabel!("Price of 100 JPY in EUR")
savefig(P, "plots/cont_prices/EUR202309_cont6_prices3.png")

P1 = select(cont_prices,Not([:JPY]))
p1 = P1[!,2:end] |> Matrix{Float64}
P2 = select(cont_prices,[:Timestamp, :JPY])
p2 = P2[!,2:end] |> Matrix{Float64}


P = plot(Time.(cont_prices.Timestamp), p1, label=permutedims(names(P1)[2:end]),legend=(0.9,0.5))
xlabel!("2023-09-22")
ylabel!("Price in EUR")
savefig(P, "plots/cont_prices/EUR202309_cont_prices1.png")

P = plot(Time.(cont_prices.Timestamp), 100 .* cont_prices.JPY, label="JPY",legend=:none)
xlabel!("2023-09-22")
ylabel!("Price of 100 JPY in EUR")
savefig(P, "plots/cont_prices/EUR202309_cont_prices3.png")


#no breaks over 6 minutes
QVplot(cont_returnsdf6.Timestamp, cont_returns6, year = false, save=false, sep=false, base="EUR", date="202309")

#all data
QVplot(returnsdf.Timestamp, returns, save=false, folder="202309",sep=false, year=false, base="EUR", date="202309")


weights = function(μ,returnsdf,returns; λ₁=1e6,λ₂=1e6)
    μ = (1+μ)^(1/(12*size(returns,1)))-1
    t = years(returnsdf.Timestamp)
    n = nrow(returnsdf)
    Δₙ = t[end] / (n-1)
    allweights = DataFrame(fill(NaN,size(returns,1),size(returns,2)+5),[quotes...,"minimum","converged", "risk","sum_weights","profit"])
    RV = C(returns)
    kₙ = k(n)
    
    for j in 1:n-kₙ
        Σ₋ = 1/(kₙ*Δₙ)*(RV[j+kₙ]-RV[j])
        μ₊ = mean([returns[i,:] for i in 1:j])
        if j%(floor(n/100)) == 0
            print("\r", round(floor((j / n) * 100),digits=2), "%")
        end
        candidate = wₜ(Σ₋,μ₊,μ, λ₁=λ₁,λ₂=λ₂)
        w = Optim.minimizer(candidate)
        f,c,r,s,p = Optim.minimum(candidate), Optim.converged(candidate), w'*Σ₋*w, (∑(w)-1), (w'*μ₊-μ)
        if !c
            @warn "Candidate did not converge at t=$t."
            #println("Trace:")
            #weights(t,μ; λ₂=1e6, trace=true)
            println("Weights:  $w, Function value: $f")
        end

        allweights[j, :] = [w...,f,c,r,s,p]
    end
    return allweights
end



plot_weights = function(timestamps,weights, quotes; avg=1, alpha=1, xlab="Date")
    #w = filter(row -> all(x -> !isnan(x), row), weights)[!,Symbol.(quotes)] |> Matrix
    w = weights[:,Symbol.(quotes)] |> Matrix
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
    xlabel!(xlab)
    ylabel!("Proportion of Portfolio")
    display(p)
end


function printweights(weights; f=[mean])
    result = ""
    header_printed = false

    for func in f
        df = mapcols(col -> func(filter(!isnan, col)), weights)
        
        if !header_printed
            result *= "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{@{}l" * repeat("r", ncol(df)) * "@{}}\n\\toprule\n"
            result *= " & " * join([" $(String(col)) " for col in names(df)], " & ") * " \\\\\n\\midrule\n"
            header_printed = true
        end

        result *= func |> string
        result *= " & " * join(["\$" * @sprintf("%.3f", df[1, j]) * "\$" for j in 1:ncol(df)], " & ")
        result *= " \\\\\n\\midrule\n"
    end

    result *= "\\bottomrule\n\\end{tabular}\n\\end{table}"
    println(result)
end


quotes, prices, p, logprices, returnsdf, returns = getquotes("EUR","202309", min_n=28000, invert=true)


calc_weights = function(μ; w = nothing, saveweights=false, f=mean, plot=false, saveplot=false, λ₁=1e6, λ₂ = 1e6)
    if isnothing(w)
        allweights = weights(μ,returnsdf,returns, λ₁=λ₁,λ₂=λ₂)

        r = cont_range(returnsdf, tol=6)
        contweights6 = weights(μ,returnsdf[r,:],returns[r,:], λ₁=λ₁,λ₂=λ₂)

        r = cont_range(returnsdf, tol=1)
        contweights = weights(μ,returnsdf[r,:],returns[r,:], λ₁=λ₁,λ₂=λ₂)

    else
        allweights, contweights6, contweights = w
    end

    println("====allweights====")
    printweights(allweights, f=functions)
    println("====contweights6====")
    printweights(contweights6, f=functions)
    println("====contweights====")
    printweights(contweights, f=functions)

    if saveweights
        @save "weights/allweights_μ=$μ.jld2" allweights
        @save "weights/contweight6s_μ=$μ.jld2" contweights6
        @save "weights/contweights_μ=$μ.jld2" contweights
    end

    if plot
        p = function(timestamps, w; r="all", xlab="Date", saveas="weights")
            if xlab == "Time"
                timestamps = Time.(timestamps)
            end
            if r=="all"
                r=1:length(timestamps)
            end
            plot_weights(timestamps[r,:],w,quotes,avg=1,xlab=xlab)
            
            if saveplot
                savefig("$(saveas)_μ=$μ.png")
            end
            plot_weights(timestamps[r,:],w,quotes,avg=60, xlab=xlab)
            if saveplot
                savefig("$(saveas)_avg60_μ=$μ.png")
            end
        end
        p(returnsdf.Timestamp, allweights, r="all",saveas="weights")
        p(returnsdf.Timestamp, contweights6, r=cont_range(returnsdf, tol=6),saveas="cont_weights_6min")
        p(returnsdf.Timestamp, contweights, r=cont_range(returnsdf, tol=1),xlab="Time",saveas="cont_weights")
    end
    if isnothing(w)
        return allweights, contweights6, contweights
    end
end

functions = [mean,maximum,minimum]
μ = 0.1
@load "weights/allweights_μ=$μ.jld2" allweights
@load "weights/contweight6s_μ=$μ.jld2" contweights6
@load "weights/contweights_μ=$μ.jld2" contweights

g = x->select(x,Not([:converged, :sum_weights, :profit]))

calc_weights(μ,w=g.([allweights,contweights6,contweights]),f=functions, plot=true, saveweights=false, saveplot=false)


μ = 0
calc_weights(μ,f=functions, plot=true, saveweights=true, saveplot=true, λ₁=1e6, λ₂=1e6)
μ = 0.1
calc_weights(μ,f=functions, plot=true, saveweights=true, saveplot=true, λ₁=1e6, λ₂=1e6)
μ = 1
calc_weights(μ,f=functions, plot=true, saveweights=true, saveplot=true, λ₁=1e6, λ₂=1e6)


μ = 0
@load "weights/allweights_μ=$μ.jld2" allweights
@load "weights/contweight6s_μ=$μ.jld2" contweights6
@load "weights/contweights_μ=$μ.jld2" contweights


allweights0 = select(allweights, Symbol.(quotes)) |> Matrix |> copy
contweights60 = select(contweights6, Symbol.(quotes)) |> Matrix |> copy
contweights0 = select(contweights, Symbol.(quotes)) |> Matrix |> copy

μ = 0.1
@load "weights/allweights_μ=$μ.jld2" allweights
@load "weights/contweight6s_μ=$μ.jld2" contweights6
@load "weights/contweights_μ=$μ.jld2" contweights

allweights = select(allweights, Symbol.(quotes)) |> Matrix
contweights6 = select(contweights6, Symbol.(quotes)) |> Matrix
contweights = select(contweights, Symbol.(quotes)) |> Matrix


plot(returnsdf.Timestamp,allweights .- allweights0, labels=permutedims(quotes))
xlabel!("Date")
ylabel!("Difference between μ=0.1 and μ=0")
savefig("allweights_0.1-0.png")
r = cont_range(returnsdf,tol=6)
plot(returnsdf.Timestamp[r],contweights6 .- contweights60, labels=permutedims(quotes))
xlabel!("Date")
ylabel!("Difference between μ=0.1 and μ=0")
savefig("contweights6_0.1-0.png")
r = cont_range(returnsdf,tol=1)
plot(Time.(returnsdf.Timestamp)[r],contweights .- contweights0, labels=permutedims(quotes))
xlabel!("Time")
ylabel!("Difference between μ=0.1 and μ=0")
savefig("contweights_0.1-0.png")



# Assuming C(returns) gives a list of 7x7 matrices
cov_matrices = C(returns)

# Calculate the mean covariance matrix
mean_cov_matrix = mean(cov_matrices)

# Plot the heatmap

heatmap(1e3.*mean_cov_matrix, color=:coolwarm, aspect_ratio=:equal, 
        xticks=(1:7, quotes), yticks=(1:7, quotes), 
        title="Mean Covariance (1000x)")

savefig("mean_cov_matrix.png")


heatmap(1e3.*cov_matrices[end], color=:coolwarm, aspect_ratio=:equal, 
xticks=(1:7, quotes), yticks=(1:7, quotes), 
title="Mean Covariance (1000x)")