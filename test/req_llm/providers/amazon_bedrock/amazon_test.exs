defmodule ReqLLM.Providers.AmazonBedrock.AmazonTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Providers.AmazonBedrock.Amazon

  @v2 "amazon.titan-embed-text-v2:0"
  @g1 "amazon.titan-embed-text-v1"
  @image "amazon.titan-embed-image-v1"
  @nova "amazon.nova-2-multimodal-embeddings-v1:0"

  describe "format_embedding_request/3" do
    test "Titan Text V2 sends inputText with dimensions and normalize" do
      assert {:ok, body} =
               Amazon.format_embedding_request(@v2, "Hello, World!",
                 dimensions: 256,
                 provider_options: [normalize: false]
               )

      assert body == %{"inputText" => "Hello, World!", "dimensions" => 256, "normalize" => false}
    end

    test "Titan Text V2 sends only inputText by default" do
      assert Amazon.format_embedding_request(@v2, "Hello, World!", []) ==
               {:ok, %{"inputText" => "Hello, World!"}}
    end

    test "Titan Text G1 takes inputText only" do
      assert Amazon.format_embedding_request(@g1, "Hello, World!", []) ==
               {:ok, %{"inputText" => "Hello, World!"}}

      assert_raise ReqLLM.Error.Invalid.Parameter, fn ->
        Amazon.format_embedding_request(@g1, "Hello, World!", dimensions: 256)
      end

      assert_raise ReqLLM.Error.Invalid.Parameter, fn ->
        Amazon.format_embedding_request(@g1, "Hello, World!",
          provider_options: [normalize: false]
        )
      end
    end

    test "Titan Multimodal embeds text with an output length" do
      assert {:ok,
              %{
                "inputText" => "Hello, World!",
                "embeddingConfig" => %{"outputEmbeddingLength" => 256}
              }} = Amazon.format_embedding_request(@image, "Hello, World!", dimensions: 256)

      assert {:ok, %{"inputText" => "Hello, World!"} = body} =
               Amazon.format_embedding_request(@image, "Hello, World!", [])

      refute Map.has_key?(body, "embeddingConfig")
    end

    test "Nova sends a single text embedding task" do
      assert {:ok, body} =
               Amazon.format_embedding_request(@nova, "Hello, World!",
                 dimensions: 1024,
                 provider_options: [
                   embedding_purpose: "GENERIC_RETRIEVAL",
                   truncation_mode: "START"
                 ]
               )

      assert body == %{
               "taskType" => "SINGLE_EMBEDDING",
               "singleEmbeddingParams" => %{
                 "embeddingPurpose" => "GENERIC_RETRIEVAL",
                 "embeddingDimension" => 1024,
                 "text" => %{"truncationMode" => "START", "value" => "Hello, World!"}
               }
             }
    end

    test "Nova defaults to GENERIC_INDEX and END truncation" do
      assert {:ok, %{"singleEmbeddingParams" => params}} =
               Amazon.format_embedding_request(@nova, "Hi", [])

      assert %{"embeddingPurpose" => "GENERIC_INDEX", "text" => %{"truncationMode" => "END"}} =
               params

      refute Map.has_key?(params, "embeddingDimension")
    end

    test "embeds one text per request" do
      assert {:ok, %{"inputText" => "Hi"}} = Amazon.format_embedding_request(@v2, ["Hi"], [])

      assert_raise ReqLLM.Error.Invalid.Parameter, ~r/one text/, fn ->
        Amazon.format_embedding_request(@v2, ["Hi", "there"], [])
      end
    end

    test "rejects other Amazon models" do
      assert_raise ReqLLM.Error.Invalid.Parameter, fn ->
        Amazon.format_embedding_request("amazon.nova-pro-v1:0", "Hi", [])
      end
    end
  end

  describe "parse_embedding_response/1" do
    test "Titan returns the vector and the token count" do
      response = %{
        "embedding" => [0.1, 0.2],
        "inputTextTokenCount" => 5,
        "embeddingsByType" => %{"float" => [0.1, 0.2]}
      }

      assert Amazon.parse_embedding_response(response) ==
               {:ok,
                %{
                  "data" => [%{"index" => 0, "embedding" => [0.1, 0.2]}],
                  "usage" => %{"prompt_tokens" => 5, "total_tokens" => 5}
                }}
    end

    test "Nova returns the text embedding" do
      response = %{"embeddings" => [%{"embeddingType" => "TEXT", "embedding" => [0.1, 0.2]}]}

      assert Amazon.parse_embedding_response(response) ==
               {:ok, %{"data" => [%{"index" => 0, "embedding" => [0.1, 0.2]}]}}
    end

    test "surfaces Titan Multimodal generation errors" do
      assert {:error, %ReqLLM.Error.API.Response{reason: reason}} =
               Amazon.parse_embedding_response(%{"message" => "Input image too large"})

      assert reason =~ "Input image too large"
    end

    test "rejects a response without a vector" do
      assert {:error, %ReqLLM.Error.API.Response{}} =
               Amazon.parse_embedding_response(%{"inputTextTokenCount" => 5})
    end
  end
end
