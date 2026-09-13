defmodule ReqLLM.Providers.AmazonBedrockPromptCacheTest do
  @moduledoc """
  Bedrock prompt caching options and their effect on Converse/native routing.
  """

  use ExUnit.Case, async: false

  alias ReqLLM.Context
  alias ReqLLM.Providers.AmazonBedrock
  alias ReqLLM.Tool

  setup do
    # Mock AWS credentials for testing
    System.put_env("AWS_ACCESS_KEY_ID", "test_key")
    System.put_env("AWS_SECRET_ACCESS_KEY", "test_secret")
    System.put_env("AWS_REGION", "us-east-1")

    context = Context.new([Context.user("test message")])

    {:ok, model} =
      ReqLLM.model(%{provider: :amazon_bedrock, id: "anthropic.claude-3-5-sonnet-20241022-v2:0"})

    {:ok, context: context, model: model}
  end

  # Helper: Determine which API was chosen based on URL
  defp get_api_type(request) do
    url_str = to_string(request.url)

    cond do
      String.contains?(url_str, "converse") -> :converse
      String.contains?(url_str, "invoke") -> :native
      true -> :unknown
    end
  end

  defp request_body(request) do
    prepared = Req.Request.prepare(request)
    Jason.decode!(prepared.body)
  end

  defp test_tool do
    Tool.new!(
      name: "test_tool",
      description: "Test",
      parameter_schema: [],
      callback: fn _ -> {:ok, "test"} end
    )
  end

  describe "routing is independent of caching" do
    test "caching with tools stays on Converse and emits cachePoint", %{
      context: context,
      model: model
    } do
      {:ok, request} =
        AmazonBedrock.prepare_request(:chat, model, context,
          tools: [test_tool()],
          provider_options: [prompt_cache: true, cache_messages: true]
        )

      assert get_api_type(request) == :converse

      body = request_body(request)

      assert List.last(body["toolConfig"]["tools"]) == %{"cachePoint" => %{"type" => "default"}}

      assert List.last(List.last(body["messages"])["content"]) == %{
               "cachePoint" => %{"type" => "default"}
             }
    end

    test "legacy alias with tools also stays on Converse", %{context: context, model: model} do
      {:ok, request} =
        AmazonBedrock.prepare_request(:chat, model, context,
          tools: [test_tool()],
          anthropic_prompt_cache: true
        )

      assert get_api_type(request) == :converse

      assert List.last(request_body(request)["toolConfig"]["tools"]) == %{
               "cachePoint" => %{"type" => "default"}
             }
    end

    test "caching without tools stays native and emits cache_control", %{model: model} do
      context = Context.new([Context.system("Stable"), Context.user("test message")])

      {:ok, request} =
        AmazonBedrock.prepare_request(:chat, model, context,
          provider_options: [prompt_cache: true, prompt_cache_ttl: "1h"]
        )

      assert get_api_type(request) == :native

      assert [%{"cache_control" => %{"type" => "ephemeral", "ttl" => "1h"}}] =
               request_body(request)["system"]
    end

    test "explicit use_converse: false with tools goes native", %{context: context, model: model} do
      {:ok, request} =
        AmazonBedrock.prepare_request(:chat, model, context,
          tools: [test_tool()],
          provider_options: [prompt_cache: true, use_converse: false]
        )

      assert get_api_type(request) == :native
      assert %{"cache_control" => _} = List.last(request_body(request)["tools"])
    end

    test "explicit use_converse: true without tools goes Converse", %{
      context: context,
      model: model
    } do
      {:ok, request} =
        AmazonBedrock.prepare_request(:chat, model, context,
          provider_options: [prompt_cache: true, use_converse: true]
        )

      assert get_api_type(request) == :converse

      assert List.last(request_body(request)["messages"]) |> Map.fetch!("content") |> List.last() ==
               %{"text" => "test message"}
    end

    test "handles empty tools list same as no tools", %{context: context, model: model} do
      {:ok, request} =
        AmazonBedrock.prepare_request(:chat, model, context,
          tools: [],
          provider_options: [prompt_cache: true]
        )

      assert get_api_type(request) == :native
    end
  end

  describe "structured output (:object) with caching" do
    @compiled_schema %{schema: %{type: "object", properties: %{}}}

    test "caches the synthetic tool on the native path", %{context: context, model: model} do
      {:ok, request} =
        AmazonBedrock.prepare_request(:object, model, context,
          compiled_schema: @compiled_schema,
          provider_options: [prompt_cache: true]
        )

      assert get_api_type(request) == :native

      assert [%{"name" => "structured_output", "cache_control" => _}] =
               request_body(request)["tools"]
    end

    test "caches the synthetic tool on Converse", %{context: context, model: model} do
      {:ok, request} =
        AmazonBedrock.prepare_request(:object, model, context,
          compiled_schema: @compiled_schema,
          provider_options: [prompt_cache: true, use_converse: true]
        )

      assert get_api_type(request) == :converse

      assert [%{"toolSpec" => %{"name" => "structured_output"}}, %{"cachePoint" => _}] =
               request_body(request)["toolConfig"]["tools"]
    end
  end

  describe "default behavior without caching" do
    test "uses native API when no tools", %{context: context, model: model} do
      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, [])
      assert get_api_type(request) == :native
    end

    test "uses Converse API when tools present", %{context: context, model: model} do
      tools = [
        Tool.new!(
          name: "test",
          description: "Test",
          parameter_schema: [],
          callback: fn _ -> {:ok, "test"} end
        )
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, tools: tools)
      assert get_api_type(request) == :converse
    end
  end

  describe "option aliases" do
    test "accepts generic options", %{model: model} do
      {:ok, opts} =
        ReqLLM.Provider.Options.process(AmazonBedrock, :chat, model,
          provider_options: [prompt_cache: true, prompt_cache_ttl: "1h", cache_messages: -2]
        )

      assert get_in(opts, [:provider_options, :prompt_cache]) == true
      assert get_in(opts, [:provider_options, :prompt_cache_ttl]) == "1h"
      assert get_in(opts, [:provider_options, :cache_messages]) == -2
    end

    test "aliases do not raise under on_unsupported: :error", %{model: model} do
      assert {:ok, _opts} =
               ReqLLM.Provider.Options.process(AmazonBedrock, :chat, model,
                 on_unsupported: :error,
                 provider_options: [anthropic_prompt_cache: true]
               )
    end

    test "rejects an unsupported ttl", %{model: model} do
      assert {:error, _} =
               ReqLLM.Provider.Options.process(AmazonBedrock, :chat, model,
                 provider_options: [prompt_cache: true, prompt_cache_ttl: "30m"]
               )
    end
  end
end
