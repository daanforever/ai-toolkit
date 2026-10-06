'use client';

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Send } from 'lucide-react';
import { apiClient } from '@/utils/api';
import { AgentChatRequest } from '@/types/agentChat';

function readChatError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data as { error?: unknown } | undefined;
    if (typeof data?.error === 'string' && data.error) {
      return data.error;
    }
  }
  return 'Request failed';
}

export default function AgentChatPanel() {
  const [draft, setDraft] = useState('');
  const [sentMessage, setSentMessage] = useState('');
  const [reply, setReply] = useState('');
  const [error, setError] = useState('');
  const [isSending, setIsSending] = useState(false);
  const sendingRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const canSend = draft.trim().length > 0 && !isSending;

  const send = async () => {
    const message = draft.trim();
    if (!message || sendingRef.current) return;
    sendingRef.current = true;
    setDraft('');
    setSentMessage(message);
    setReply('');
    setError('');
    setIsSending(true);
    try {
      const body: AgentChatRequest = { message };
      const res = await apiClient.post<{ reply?: unknown }>('/api/agent/chat', body);
      if (!mountedRef.current) return;
      if (typeof res.data?.reply !== 'string') {
        setError('Empty reply');
        return;
      }
      setReply(res.data.reply);
    } catch (requestError) {
      if (!mountedRef.current) return;
      setError(readChatError(requestError));
    } finally {
      sendingRef.current = false;
      if (mountedRef.current) setIsSending(false);
    }
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void send();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  };

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-l border-gray-700 bg-gray-950">
      <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-3 text-sm">
        {sentMessage && (
          <p className="whitespace-pre-wrap break-words rounded-lg bg-gray-800 px-3 py-2 text-gray-200">
            {sentMessage}
          </p>
        )}
        {isSending && <p className="text-gray-500">Thinking…</p>}
        {reply && <p className="whitespace-pre-wrap break-words text-gray-200">{reply}</p>}
        {error && <p className="whitespace-pre-wrap break-words text-red-400">{error}</p>}
      </div>
      <form className="shrink-0 border-t border-gray-700 p-3" onSubmit={onSubmit}>
        <div className="relative">
          <textarea
            aria-label="Message"
            rows={3}
            placeholder="Message"
            value={draft}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={onKeyDown}
            className="w-full resize-none rounded-lg border border-gray-700 bg-gray-900 py-2 pl-3 pr-10 text-sm text-gray-200 outline-none focus:ring-2 focus:ring-gray-600"
          />
          <button
            type="submit"
            disabled={!canSend}
            aria-label="Send"
            className="absolute bottom-2 right-2 rounded-md p-1 text-gray-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </form>
    </aside>
  );
}
