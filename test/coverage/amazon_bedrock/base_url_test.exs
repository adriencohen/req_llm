defmodule ReqLLM.Coverage.AmazonBedrock.BaseUrlTest do
  @moduledoc """
  Record with REQ_LLM_FIXTURES_MODE=record.
  """
  use ExUnit.Case, async: false

  import ReqLLM.Test.Helpers

  @moduletag :coverage
  @moduletag provider: "amazon_bedrock"
  @moduletag timeout: 180_000

  test "sends the request to a custom base_url" do
    opts =
      fixture_opts("fips_base_url",
        max_tokens: 64,
        base_url: "https://bedrock-runtime-fips.us-east-1.amazonaws.com",
        provider_options: [region: "us-east-1"]
      )

    {:ok, response} =
      ReqLLM.generate_text("amazon_bedrock:amazon.nova-lite-v1:0", "Say hi in two words.", opts)

    assert ReqLLM.Response.text(response) =~ ~r/\S/
  end
end
