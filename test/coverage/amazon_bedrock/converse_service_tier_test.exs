defmodule ReqLLM.Coverage.AmazonBedrock.ConverseServiceTierTest do
  @moduledoc """
  Record with REQ_LLM_FIXTURES_MODE=record and AWS_REGION=us-east-1.
  """
  use ExUnit.Case, async: false

  import ReqLLM.Test.Helpers

  @moduletag :coverage
  @moduletag provider: "amazon_bedrock"
  @moduletag timeout: 180_000

  @model "amazon_bedrock:openai.gpt-oss-20b-1:0"

  test "runs on the flex service tier through Converse" do
    opts =
      fixture_opts("converse_service_tier_flex",
        max_tokens: 400,
        service_tier: "flex",
        provider_options: [use_converse: true]
      )

    {:ok, response} = ReqLLM.generate_text(@model, "Say hi in two words.", opts)

    assert ReqLLM.Response.text(response) =~ ~r/\w/
  end
end
