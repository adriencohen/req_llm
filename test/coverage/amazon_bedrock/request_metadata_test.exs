defmodule ReqLLM.Coverage.AmazonBedrock.RequestMetadataTest do
  @moduledoc """
  Record with REQ_LLM_FIXTURES_MODE=record.
  """
  use ExUnit.Case, async: false

  import ReqLLM.Test.Helpers

  @moduletag :coverage
  @moduletag provider: "amazon_bedrock"
  @moduletag timeout: 180_000

  @model "amazon_bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0"

  test "sends request metadata on Converse" do
    opts =
      fixture_opts("converse_request_metadata",
        max_tokens: 64,
        provider_options: [region: "eu-west-1", request_metadata: %{tenant: "acme"}]
      )

    {:ok, response} = ReqLLM.generate_text(@model, "Say hi in two words.", opts)

    assert ReqLLM.Response.text(response) =~ ~r/\S/
  end
end
