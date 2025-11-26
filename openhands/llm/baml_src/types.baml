// Message content types
class TextContent {
  type string
  text string
  cache_control map<string, string>?
}

class ImageContent {
  type string
  image_url map<string, string>
  cache_control map<string, string>?
}

// Union type for content
type Content = TextContent | ImageContent

// Message class matching OpenHands Message structure
class Message {
  role "user" | "system" | "assistant" | "tool"
  content Content[]
  tool_calls map<string, string>[]?
  tool_call_id string?
  name string?
}

// Tool definition for function calling
class Tool {
  type string
  function_def map<string, string>
}

// LLM completion request
class LLMCompletionRequest {
  messages Message[]
  tools Tool[]?
  temperature float?
  max_tokens int?
  top_p float?
  top_k float?
  seed int?
  stop string[]?
}

// Response structure matching ModelResponse
class Choice {
  index int
  message Message
  finish_reason string?
}

class Usage {
  prompt_tokens int?
  completion_tokens int?
  total_tokens int?
  prompt_tokens_details map<string, string>?
}

class LLMCompletionResponse {
  id string
  choices Choice[]
  usage Usage?
  model string?
  created int?
}

