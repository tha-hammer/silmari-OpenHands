// Main completion function for LLM requests
// This function accepts formatted conversation text and extracts the response
function CompleteLLMRequest(request: FormattedLLMRequest) -> LLMCompletionResponse {
  // Use LiteLLMClient which is configured via environment variables
  // BAML_API_KEY, BAML_BASE_URL, BAML_MODEL are set from LLMConfig
  client LiteLLMClient

  prompt #"
    {{ request.conversation }}

    Please respond to the conversation above.

    {{ ctx.output_format }}
  "#
}

