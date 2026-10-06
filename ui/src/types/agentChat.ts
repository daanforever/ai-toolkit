export interface AgentChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface AgentChatRequest {
  messages: AgentChatTurn[];
  imgPath?: string;
  caption?: string;
}
