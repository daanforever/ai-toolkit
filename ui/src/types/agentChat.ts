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
  reasoning?: boolean;
}

export type AgentChatStreamEvent =
  | { type: 'reasoning'; delta: string }
  | { type: 'content'; delta: string }
  | { type: 'error'; message: string }
  | { type: 'edit'; caption: string };
