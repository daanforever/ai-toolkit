export interface AgentChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface AgentChatDisplayTurn extends AgentChatTurn {
  reasoning?: string;
}

export interface AgentChatRequest {
  messages: AgentChatTurn[];
  imgPath?: string;
  caption?: string;
}

export type AgentChatStreamEvent =
  | { type: 'reasoning'; delta: string }
  | { type: 'content'; delta: string }
  | { type: 'error'; message: string };
