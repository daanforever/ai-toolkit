'use client';

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Send, Square } from 'lucide-react';
import { apiClient } from '@/utils/api';
import { AgentChatRequest, AgentChatTurn } from '@/types/agentChat';

interface Props {
  imgPath: string | null;
  caption: string | null;
}

interface AttachmentSnapshot {
  imgPath?: string;
  caption?: string;
}

interface PendingSend {
  controller: AbortController;
  message: string;
  priorTurns: AgentChatTurn[];
  isOpening: boolean;
}

function readChatError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data as { error?: unknown } | undefined;
    if (typeof data?.error === 'string' && data.error) {
      return data.error;
    }
  }
  return 'Request failed';
}

function captureAttachment(imgPath: string | null, caption: string | null): AttachmentSnapshot {
  const snapshot: AttachmentSnapshot = {};
  if (imgPath) snapshot.imgPath = imgPath;
  const trimmedCaption = caption?.trim() ?? '';
  if (trimmedCaption) snapshot.caption = trimmedCaption;
  return snapshot;
}

function captionNotice(typed: string, reflected: string, loaded: string): string {
  if (loaded === reflected) return typed;
  const notice = loaded ? `Caption changed to:\n${loaded}` : 'Caption removed.';
  return `${notice}\n\n${typed}`;
}

export default function AgentChatPanel({ imgPath, caption }: Props) {
  const [draft, setDraft] = useState('');
  const [turns, setTurns] = useState<AgentChatTurn[]>([]);
  const [attachment, setAttachment] = useState<AttachmentSnapshot | null>(null);
  const [reflectedCaption, setReflectedCaption] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [isSending, setIsSending] = useState(false);
  const sendingRef = useRef(false);
  const mountedRef = useRef(false);
  const pendingRef = useRef<PendingSend | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      pendingRef.current?.controller.abort();
    };
  }, []);

  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    list.scrollTop = list.scrollHeight;
  }, [turns, isSending, error]);

  const canSend = draft.trim().length > 0 && !isSending;

  const rollback = (pending: PendingSend, errorText = '') => {
    if (!mountedRef.current) return;
    setTurns(pending.priorTurns);
    if (pending.isOpening) setAttachment(null);
    setError(errorText);
    setDraft(current => (current.length > 0 ? current : pending.message));
  };

  const finishPending = (pending: PendingSend) => {
    if (pendingRef.current !== pending) return;
    pendingRef.current = null;
    sendingRef.current = false;
    if (mountedRef.current) setIsSending(false);
  };

  const cancel = () => {
    const pending = pendingRef.current;
    if (!pending) return;
    pendingRef.current = null;
    sendingRef.current = false;
    pending.controller.abort();
    rollback(pending);
    if (mountedRef.current) setIsSending(false);
  };

  const send = async () => {
    const message = draft.trim();
    if (!message || sendingRef.current || !mountedRef.current) return;

    const pending: PendingSend = {
      controller: new AbortController(),
      message,
      priorTurns: turns,
      isOpening: turns.length === 0,
    };
    const { controller, priorTurns, isOpening } = pending;
    const signal = controller.signal;
    pendingRef.current = pending;
    sendingRef.current = true;

    const snapshot = isOpening ? captureAttachment(imgPath, caption) : (attachment ?? {});
    const loadedCaption = typeof caption === 'string' ? caption.trim() : null;
    const content =
      !isOpening && reflectedCaption !== null && loadedCaption !== null
        ? captionNotice(message, reflectedCaption, loadedCaption)
        : message;
    const userTurn: AgentChatTurn = { role: 'user', content };
    const nextTurns: AgentChatTurn[] = [...priorTurns, userTurn];

    setDraft('');
    setError('');
    setTurns(nextTurns);
    if (isOpening) setAttachment(snapshot);
    setIsSending(true);

    const isCurrent = () => mountedRef.current && pendingRef.current === pending;

    try {
      const body: AgentChatRequest = { messages: nextTurns };
      if (snapshot.imgPath) body.imgPath = snapshot.imgPath;
      if (snapshot.caption) body.caption = snapshot.caption;
      const res = await apiClient.post<{ reply?: unknown }>('/api/agent/chat', body, { signal });
      if (!isCurrent()) return;
      if (signal.aborted) {
        rollback(pending);
        return;
      }
      const reply = res.data?.reply;
      if (typeof reply !== 'string' || reply.trim() === '') {
        rollback(pending, 'Empty reply');
        return;
      }
      setTurns([...nextTurns, { role: 'assistant', content: reply }]);
      if (isOpening) setReflectedCaption(snapshot.caption ?? '');
      else if (loadedCaption !== null) setReflectedCaption(loadedCaption);
    } catch (requestError) {
      if (!isCurrent()) return;
      rollback(pending, signal.aborted ? '' : readChatError(requestError));
    } finally {
      finishPending(pending);
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
      <div ref={listRef} className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-3 text-sm">
        {turns.map((turn, index) =>
          turn.role === 'user' ? (
            <p key={index} className="whitespace-pre-wrap break-words rounded-lg bg-gray-800 px-3 py-2 text-gray-200">
              {turn.content}
            </p>
          ) : (
            <p key={index} className="whitespace-pre-wrap break-words text-gray-200">
              {turn.content}
            </p>
          ),
        )}
        {isSending && <p className="text-gray-500">Thinking…</p>}
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
            type={isSending ? 'button' : 'submit'}
            disabled={!isSending && !canSend}
            aria-label={isSending ? 'Cancel' : 'Send'}
            onClick={isSending ? cancel : undefined}
            className="absolute bottom-2 right-2 rounded-md p-1 text-gray-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isSending ? <Square className="h-4 w-4" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
      </form>
    </aside>
  );
}
