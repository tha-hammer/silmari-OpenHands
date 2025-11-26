// Simplified message format for BAML
// The adapter will format messages to text before calling this function
class FormattedMessage {
  role "user" | "system" | "assistant" | "tool"
  content string
}

// Simplified request that accepts formatted messages as text
class FormattedLLMRequest {
  conversation string
  temperature float?
  max_tokens int?
  top_p float?
  top_k float?
  seed int?
  stop string[]?
}

// Main completion function for LLM requests
// This function accepts formatted conversation text and extracts the response
function CompleteLLMRequest(request: FormattedLLMRequest) -> LLMCompletionResponse {
  client LiteLLMClient

  prompt #"
    {{ request.conversation }}

    Please respond to the conversation above.

    {{ ctx.output_format }}
  "#
}

