using Random, XLSX, LinearAlgebra, Optim, ShiftedArrays, Statistics, Plots, JLD2, DataFrames, Dates, Printf
Random.seed!(3)
∑ = x->sum(x)

function W(n, μ, σ, X₀, Δₙ)
    μ, σ, X₀, Δₙ = float([μ, σ, X₀, Δₙ])
    ΔX = μ * Δₙ .+ σ * sqrt(Δₙ) * randn(n)  # Change in log price
    return [X₀, X₀ .+ cumsum(ΔX)...]
end

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

# Parameters
n = Int(30 * 24 * 60) # Assuming 30 trading days and 24 hours of trading per day  
μ = 0.1  # Annual drift rate
σ = 1  # Annual volatility 
T = 30/365  # Annual time horizon
X₀ = 10 #Initial price
Δₙ = T/n #Discretization coefficient
d = 5   # Number of quotes

logprices = DataFrame(hcat([W(n, μ, σ, X₀, Δₙ) for _ in 1:d]...), :auto)

quotes = names(logprices)

returnsdf = DataFrame(zeros(n,d),quotes)

for q in quotes
    returnsdf[!, Symbol(q)] = diff(logprices[!,Symbol(q)])
end

returns = returnsdf |> Matrix

quadratic_variation = cumsum(returns.^2,dims=1)
plot(range(0,30,n),quadratic_variation/T, legend=:none) #t
xlabel!("Time in days")
ylabel!("Quadratic variation")

savefig("plots/fake_qv.png")

weights = function(μ_year,Δₙ,returns; λ₁=1e6,λ₂=1e6)
    μ = (1+μ_year)^(1/(12*size(returns,1)))-1
    n = nrow(returnsdf)
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

plot_weights = function(timestamps,weights, quotes; avg=1, alpha=1)
    #w = filter(row -> all(x -> !isnan(x), row), weights)[!,Symbol.(quotes)] |> Matrix
    w = weights[!,Symbol.(quotes)] |> Matrix
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

μ_year = 0
allweights = weights(μ_year,Δₙ,returns,λ₁=1e6,λ₂=1e6)
@save "weights/fakeweights_μ=$μ_year.jld2" allweights

functions = [mean, minimum, maximum]
printweights(select(allweights, Not([:converged])), f=functions)





@load "weights/fakeweights_μ=$μ_year.jld2" allweights


plot_weights(range(0,30,n),allweights,quotes,avg=1)
xlabel!("Date")
ylabel!("Proportion of Portfolio")
savefig("fakeweights_μ=$μ_year.png")


