defmodule ReqLLM.Coverage.AmazonBedrock.MantleGpt6Test do
  @moduledoc """
  Record with REQ_LLM_FIXTURES_MODE=record.
  """
  use ExUnit.Case, async: false

  import ReqLLM.Test.Helpers

  @moduletag :coverage
  @moduletag provider: "amazon_bedrock"
  @moduletag timeout: 180_000

  @model "amazon_bedrock:openai.gpt-6.1-sol"
  @opts [max_tokens: 64, provider_options: [endpoint: :mantle, region: "us-east-1"]]

  test "answers on the /openai/v1 Responses route" do
    {:ok, response} =
      ReqLLM.generate_text(
        @model,
        "Say hi in two words.",
        fixture_opts("mantle_responses", @opts)
      )

    assert ReqLLM.Response.text(response) =~ ~r/\S/
    assert response.usage.output_tokens > 0
  end

  test "runs on the ultrafast service tier" do
    opts = fixture_opts("mantle_ultrafast", Keyword.put(@opts, :service_tier, "ultrafast"))

    {:ok, response} = ReqLLM.generate_text(@model, "Say hi in two words.", opts)

    assert ReqLLM.Response.text(response) =~ ~r/\S/
    assert response.provider_meta["service_tier"] == "ultrafast"
  end

  test "returns a reasoning summary" do
    opts =
      fixture_opts(
        "mantle_reasoning_summary",
        max_tokens: 2000,
        reasoning_effort: :high,
        provider_options: [endpoint: :mantle, region: "us-east-1", reasoning_summary: "detailed"]
      )

    {:ok, response} =
      ReqLLM.generate_text(
        @model,
        "How many positive integers below 500 are divisible by 3 or 7 but not by 21? Answer with the number only.",
        opts
      )

    assert ReqLLM.Response.thinking(response) =~ ~r/\S/
  end
end
