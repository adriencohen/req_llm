defmodule ReqLLM.Coverage.AmazonBedrock.RuntimeCompatibleTest do
  @moduledoc """
  Record with REQ_LLM_FIXTURES_MODE=record.
  """
  use ExUnit.Case, async: false

  import ReqLLM.Test.Helpers

  @moduletag :coverage
  @moduletag provider: "amazon_bedrock"
  @moduletag timeout: 180_000

  @routes [
    chat_completions: "amazon_bedrock:openai.gpt-oss-120b-1:0",
    messages: "amazon_bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0"
  ]

  @opts [max_tokens: 400, provider_options: [region: "eu-west-1", runtime_api: :compatible]]

  for {route, model} <- @routes do
    @model model

    test "#{route}: generate_text on bedrock-runtime" do
      opts = fixture_opts("runtime_compatible_basic", @opts)

      {:ok, response} = ReqLLM.generate_text(@model, "Say hi in two words.", opts)

      assert ReqLLM.Response.text(response) =~ ~r/\S/
      assert response.usage.output_tokens > 0
    end

    test "#{route}: stream_text on bedrock-runtime" do
      opts = fixture_opts("runtime_compatible_streaming", @opts)

      {:ok, response} = ReqLLM.stream_text(@model, "Say hi in two words.", opts)

      assert ReqLLM.StreamResponse.text(response) =~ ~r/\S/
    end
  end
end
