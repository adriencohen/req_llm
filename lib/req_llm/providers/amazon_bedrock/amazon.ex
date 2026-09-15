defmodule ReqLLM.Providers.AmazonBedrock.Amazon do
  @moduledoc """
  Amazon Titan and Nova embedding models on AWS Bedrock, one text per request.

  See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html,
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-mm.html
  and https://docs.aws.amazon.com/nova/latest/userguide/nova-embeddings.html
  """

  alias ReqLLM.Error

  @doc "Build the InvokeModel body for a Titan or Nova embedding model."
  def format_embedding_request(model_id, [text], opts),
    do: format_embedding_request(model_id, text, opts)

  def format_embedding_request(_model_id, texts, _opts) when is_list(texts) do
    raise Error.Invalid.Parameter,
      parameter: "Amazon embedding models embed one text per request, got #{length(texts)}"
  end

  def format_embedding_request(model_id, text, opts) when is_binary(text) do
    provider_opts = opts[:provider_options] || []
    dimensions = opts[:dimensions] || provider_opts[:dimensions]

    {:ok, body(model_id, text, dimensions, provider_opts)}
  end

  defp body("amazon.nova-2-multimodal-embeddings-v1:0", text, dimensions, provider_opts),
    do: nova(text, dimensions, provider_opts)

  defp body("amazon.titan-embed-image-v1", text, dimensions, _provider_opts),
    do: titan_image(text, dimensions)

  defp body("amazon.titan-embed-text-v2:0", text, dimensions, provider_opts),
    do: titan_v2(text, dimensions, provider_opts)

  defp body(model_id, text, dimensions, provider_opts)
       when model_id in ["amazon.titan-embed-text-v1", "amazon.titan-embed-g1-text-02"],
       do: titan_g1(text, dimensions, provider_opts)

  defp body(model_id, _text, _dimensions, _provider_opts),
    do: raise(Error.Invalid.Parameter, parameter: "#{model_id} is not an Amazon embedding model")

  defp titan_v2(text, dimensions, provider_opts) do
    %{"inputText" => text}
    |> put_option("dimensions", dimensions)
    |> put_option("normalize", provider_opts[:normalize])
  end

  defp titan_g1(text, nil, provider_opts) do
    if Keyword.has_key?(provider_opts, :normalize),
      do:
        raise(Error.Invalid.Parameter, parameter: "Titan Embeddings G1 takes no normalize option")

    %{"inputText" => text}
  end

  defp titan_g1(_text, _dimensions, _provider_opts),
    do:
      raise(Error.Invalid.Parameter, parameter: "Titan Embeddings G1 takes no dimensions option")

  defp titan_image(text, nil), do: %{"inputText" => text}

  defp titan_image(text, dimensions),
    do: %{"inputText" => text, "embeddingConfig" => %{"outputEmbeddingLength" => dimensions}}

  defp nova(text, dimensions, provider_opts) do
    params =
      %{
        "embeddingPurpose" => Keyword.get(provider_opts, :embedding_purpose, "GENERIC_INDEX"),
        "text" => %{
          "truncationMode" => Keyword.get(provider_opts, :truncation_mode, "END"),
          "value" => text
        }
      }
      |> put_option("embeddingDimension", dimensions)

    %{"taskType" => "SINGLE_EMBEDDING", "singleEmbeddingParams" => params}
  end

  defp put_option(body, _key, nil), do: body
  defp put_option(body, key, value), do: Map.put(body, key, value)

  @doc "Normalize a Titan or Nova response to the `data` list `ReqLLM.embed/3` reads."
  def parse_embedding_response(%{"message" => message}) when is_binary(message),
    do: {:error, Error.API.Response.exception(reason: "Amazon embedding error: #{message}")}

  def parse_embedding_response(%{"embedding" => [_ | _] = embedding} = response) do
    {:ok, put_usage(data(embedding), response["inputTextTokenCount"])}
  end

  def parse_embedding_response(%{"embeddings" => [%{"embedding" => [_ | _] = embedding}]}),
    do: {:ok, data(embedding)}

  def parse_embedding_response(response) do
    {:error,
     Error.API.Response.exception(
       reason: "Amazon embedding response has no vector",
       response_body: response
     )}
  end

  defp data(embedding), do: %{"data" => [%{"index" => 0, "embedding" => embedding}]}

  defp put_usage(body, count) when is_integer(count),
    do: Map.put(body, "usage", %{"prompt_tokens" => count, "total_tokens" => count})

  defp put_usage(body, _count), do: body
end
