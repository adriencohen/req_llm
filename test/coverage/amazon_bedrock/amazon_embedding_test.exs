defmodule ReqLLM.Coverage.AmazonBedrock.AmazonEmbeddingTest do
  @moduledoc """
  Record with REQ_LLM_FIXTURES_MODE=record and AWS_REGION=us-east-1.
  """
  use ExUnit.Case, async: false

  import ReqLLM.Test.Helpers

  @moduletag :coverage
  @moduletag provider: "amazon_bedrock"
  @moduletag category: :embedding
  @moduletag timeout: 180_000

  @titan_v2 "amazon_bedrock:amazon.titan-embed-text-v2:0"
  @titan_g1 "amazon_bedrock:amazon.titan-embed-text-v1"
  @titan_image "amazon_bedrock:amazon.titan-embed-image-v1"
  @nova "amazon_bedrock:amazon.nova-2-multimodal-embeddings-v1:0"

  setup_all do
    LLMDB.load(allow: :all, custom: %{})
    :ok
  end

  test "Titan Text V2 embeds with usage" do
    {:ok, %{embedding: embedding, usage: usage}} =
      ReqLLM.embed(@titan_v2, "Hello, World!", fixture_opts("embed_basic", return_usage: true))

    assert length(embedding) == 1024
    assert usage.input > 0
  end

  test "Titan Text V2 honors dimensions" do
    {:ok, embedding} =
      ReqLLM.embed(@titan_v2, "Hello, World!", fixture_opts("embed_256", dimensions: 256))

    assert length(embedding) == 256
  end

  test "Titan Text G1 embeds" do
    {:ok, embedding} = ReqLLM.embed(@titan_g1, "Hello, World!", fixture_opts("embed_basic"))

    assert length(embedding) == 1536
  end

  test "Titan Multimodal embeds text" do
    {:ok, embedding} =
      ReqLLM.embed(@titan_image, "Hello, World!", fixture_opts("embed_256", dimensions: 256))

    assert length(embedding) == 256
  end

  test "Nova embeds with usage" do
    {:ok, %{embedding: embedding, usage: usage}} =
      ReqLLM.embed(
        @nova,
        "Hello, World!",
        fixture_opts("embed_1024", dimensions: 1024, return_usage: true)
      )

    assert length(embedding) == 1024
    assert usage.input > 0
  end
end
