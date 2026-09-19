defmodule ReqLLM.Coverage.AmazonBedrock.ReasoningEffortTest do
  @moduledoc """
  Record with REQ_LLM_FIXTURES_MODE=record and AWS_REGION=us-east-1. The Claude and
  Nova tests pin eu-west-1, where their cross-region profiles are callable.
  """
  use ExUnit.Case, async: false

  import ReqLLM.Test.Helpers

  @moduletag :coverage
  @moduletag provider: "amazon_bedrock"
  @moduletag timeout: 180_000

  @model "amazon_bedrock:openai.gpt-oss-20b-1:0"
  @claude "amazon_bedrock:eu.anthropic.claude-sonnet-5"
  @nova "amazon_bedrock:eu.amazon.nova-2-lite-v1:0"
  @prompt "What is 17 times 23? Answer with the number only."
  @opts [max_tokens: 4096, reasoning_effort: :high]
  @eu_opts [region: "eu-west-1", reasoning_effort: :low]

  setup_all do
    LLMDB.load(allow: :all, custom: %{})
    :ok
  end

  test "gpt-oss reasons at the requested effort on InvokeModel" do
    {:ok, response} =
      ReqLLM.generate_text(@model, @prompt, fixture_opts("reasoning_effort_invoke", @opts))

    text = ReqLLM.Response.text(response)

    assert text =~ "<reasoning>"
    assert text =~ "391"
  end

  test "gpt-oss reasons at the requested effort on Converse" do
    opts = @opts ++ [provider_options: [use_converse: true]]

    {:ok, response} =
      ReqLLM.generate_text(@model, @prompt, fixture_opts("reasoning_effort_converse", opts))

    assert ReqLLM.Response.thinking(response) =~ ~r/\S/
    assert ReqLLM.Response.text(response) =~ "391"
  end

  test "adaptive Claude takes the effort on InvokeModel" do
    opts = @eu_opts ++ [max_tokens: 1024]

    {:ok, response} =
      ReqLLM.generate_text(@claude, @prompt, fixture_opts("reasoning_effort_invoke", opts))

    assert ReqLLM.Response.text(response) =~ "391"
  end

  test "adaptive Claude takes the effort on Converse" do
    opts = @eu_opts ++ [max_tokens: 1024, provider_options: [use_converse: true]]

    {:ok, response} =
      ReqLLM.generate_text(@claude, @prompt, fixture_opts("reasoning_effort_converse", opts))

    assert ReqLLM.Response.text(response) =~ "391"
  end

  test "Nova 2 reasons at the requested effort on Converse" do
    opts = @eu_opts ++ [max_tokens: 4096]

    {:ok, response} =
      ReqLLM.generate_text(@nova, @prompt, fixture_opts("reasoning_effort_converse", opts))

    assert ReqLLM.Response.thinking(response) =~ ~r/\S/
    assert ReqLLM.Response.text(response) =~ "391"
  end
end
