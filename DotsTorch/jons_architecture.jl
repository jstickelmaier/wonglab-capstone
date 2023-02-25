	struct VariationalAutoencoder
		encoder
		μ
		logσ
		decoder
	end

	Flux.@functor VariationalAutoencoder (encoder, μ, logσ, decoder)

	function (model::VariationalAutoencoder)(x)
		(model.μ ∘ model.encoder)(x)
	end

	function forward(model::VariationalAutoencoder, x)
		h = model.encoder(x)
		μ, logσ = model.μ(h), model.logσ(h)
		z = μ + randn(Float32, size(μ)) .* exp.(logσ)
		μ, logσ, model.decoder(z)
	end
	
	function loss(model::VariationalAutoencoder, x; β = 1f-1)
		μ, logσ, x̂ = forward(model, x)
	
	  	reconstruction_loss = Flux.mse(x, x̂)
		regularization_loss = 0.5f0 * mean(μ.^2 .+ exp.(logσ).^2 - 2f0 * logσ .- 1f0) ./ size(μ, 2)

	  	reconstruction_loss + β * regularization_loss
	end