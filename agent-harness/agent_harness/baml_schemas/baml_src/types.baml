// Types for LLM completion requests and responses

// Simplified message format for BAML
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

// Response from LLM completion
class LLMCompletionResponse {
  content string
  tool_calls map<string, string>?
  prompt_tokens int?
  completion_tokens int?
  total_tokens int?
}

