'use client';

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import { Send, Square } from 'lucide-react';
import { authHeaders, clearUnauthorized } from '@/utils/api';
import { AgentChatDisplayTurn, AgentChatRequest, AgentChatStreamEvent } from '@/types/agentChat';

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
  priorTurns: AgentChatDisplayTurn[];
  isOpening: boolean;
}

interface StreamPartial {
  reasoning: string;
  content: string;
}

type StreamOutcome =
  | ({ status: 'ok' } & StreamPartial)
  | { status: 'error'; message: string }
  | { status: 'aborted' };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseStreamLine(line: string): AgentChatStreamEvent | null {
  let value: unknown;
  try {
    value = JSON.parse(line);
  } catch (error) {
    console.error('Agent chat stream line was invalid JSON:', error);
    return null;
  }
  if (!isRecord(value)) {
    console.error('Agent chat stream line was invalid:', value);
    return null;
  }
  if ((value.type === 'reasoning' || value.type === 'content') && typeof value.delta === 'string') {
    return { type: value.type, delta: value.delta };
  }
  if (value.type === 'error' && typeof value.message === 'string') {
    return { type: 'error', message: value.message };
  }
  console.error('Agent chat stream line was invalid:', value);
  return null;
}

function takeLines(buffer: string): { lines: string[]; rest: string } {
  const lines = buffer.split('\n');
  const rest = lines.pop() ?? '';
  return { lines, rest };
}

async function readErrorResponse(response: Response): Promise<string> {
  try {
    const data: unknown = await response.json();
    if (isRecord(data) && typeof data.error === 'string' && data.error) return data.error;
  } catch (error) {
    console.error('Agent chat error response was not JSON:', error);
  }
  return 'Request failed';
}

async function readAgentChatStream(
  response: Response,
  signal: AbortSignal,
  onPartial: (partial: StreamPartial) => void,
): Promise<StreamOutcome> {
  const body = response.body;
  if (!body) return { status: 'error', message: 'Request failed' };

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const partial: StreamPartial = { reasoning: '', content: '' };
  let changed = false;

  const fail = (message: string): StreamOutcome => ({ status: 'error', message });

  const applyLine = (line: string): StreamOutcome | null => {
    const trimmed = line.replace(/\r$/, '').trim();
    if (!trimmed) return null;
    const event = parseStreamLine(trimmed);
    if (!event) return fail('Request failed');
    if (event.type === 'error') return fail(event.message || 'Request failed');
    if (event.type === 'reasoning') partial.reasoning += event.delta;
    else partial.content += event.delta;
    changed = true;
    return null;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (signal.aborted) return { status: 'aborted' };
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parsed = takeLines(buffer);
      buffer = parsed.rest;
      for (const line of parsed.lines) {
        const outcome = applyLine(line);
        if (outcome) return outcome;
      }
      if (changed) {
        changed = false;
        onPartial({ reasoning: partial.reasoning, content: partial.content });
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      const outcome = applyLine(buffer);
      if (outcome) return outcome;
      if (changed) onPartial({ reasoning: partial.reasoning, content: partial.content });
    }
    return { status: 'ok', reasoning: partial.reasoning, content: partial.content };
  } catch (error) {
    if (signal.aborted) return { status: 'aborted' };
    console.error('Agent chat request failed:', error);
    return fail('Request failed');
  } finally {
    try {
      await reader.cancel();
    } catch (error) {
      if (!signal.aborted) console.error('Agent chat stream cancel failed:', error);
    }
  }
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

function ThinkingBlock({ reasoning, expanded = false }: { reasoning: string; expanded?: boolean }) {
  const body = (
    <>
      <summary className="cursor-pointer">Thinking</summary>
      <p className="mt-1 whitespace-pre-wrap break-words">{reasoning}</p>
    </>
  );
  if (expanded) {
    return <details open className="text-gray-500">{body}</details>;
  }
  return <details className="text-gray-500">{body}</details>;
}

export default function AgentChatPanel({ imgPath, caption }: Props) {
  const [draft, setDraft] = useState('');
  const [turns, setTurns] = useState<AgentChatDisplayTurn[]>([]);
  const [streaming, setStreaming] = useState<StreamPartial | null>(null);
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
  }, [turns, isSending, error, streaming]);

  const canSend = draft.trim().length > 0 && !isSending;
  const waiting = isSending && !streaming?.reasoning && !streaming?.content;

  const rollback = (pending: PendingSend, errorText = '') => {
    if (!mountedRef.current) return;
    setTurns(pending.priorTurns);
    if (pending.isOpening) setAttachment(null);
    setStreaming(null);
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
    const userTurn: AgentChatDisplayTurn = { role: 'user', content };
    const nextTurns: AgentChatDisplayTurn[] = [...priorTurns, userTurn];

    setDraft('');
    setError('');
    setStreaming(null);
    setTurns(nextTurns);
    if (isOpening) setAttachment(snapshot);
    setIsSending(true);

    const isCurrent = () => mountedRef.current && pendingRef.current === pending;

    try {
      const body: AgentChatRequest = {
        messages: nextTurns.map(({ role, content: turnContent }) => ({ role, content: turnContent })),
      };
      if (snapshot.imgPath) body.imgPath = snapshot.imgPath;
      if (snapshot.caption) body.caption = snapshot.caption;
      const response = await fetch('/api/agent/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders(),
        },
        body: JSON.stringify(body),
        signal,
      });
      if (!isCurrent()) return;
      clearUnauthorized(response.status);
      if (signal.aborted) {
        rollback(pending);
        return;
      }
      if (!response.ok) {
        const messageText = await readErrorResponse(response);
        if (!isCurrent()) return;
        rollback(pending, signal.aborted ? '' : messageText);
        return;
      }

      const outcome = await readAgentChatStream(response, signal, partial => {
        if (isCurrent()) setStreaming(partial);
      });
      if (!isCurrent()) return;
      if (signal.aborted || outcome.status === 'aborted') {
        rollback(pending);
        return;
      }
      if (outcome.status === 'error') {
        rollback(pending, outcome.message);
        return;
      }
      if (outcome.content.trim() === '') {
        rollback(pending, 'Empty reply');
        return;
      }

      const assistantTurn: AgentChatDisplayTurn = { role: 'assistant', content: outcome.content };
      if (outcome.reasoning.trim()) assistantTurn.reasoning = outcome.reasoning;
      setStreaming(null);
      setTurns([...nextTurns, assistantTurn]);
      if (isOpening) setReflectedCaption(snapshot.caption ?? '');
      else if (loadedCaption !== null) setReflectedCaption(loadedCaption);
    } catch (requestError) {
      if (!isCurrent()) return;
      if (!signal.aborted) console.error('Agent chat request failed:', requestError);
      rollback(pending, signal.aborted ? '' : 'Request failed');
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
            <div key={index} className="flex flex-col gap-3">
              {turn.reasoning ? <ThinkingBlock reasoning={turn.reasoning} /> : null}
              <p className="whitespace-pre-wrap break-words text-gray-200">{turn.content}</p>
            </div>
          ),
        )}
        {streaming?.reasoning ? (
          <ThinkingBlock
            key={streaming.content.length > 0 ? 'settled' : 'live'}
            reasoning={streaming.reasoning}
            expanded={streaming.content.length === 0}
          />
        ) : null}
        {streaming?.content ? (
          <p className="whitespace-pre-wrap break-words text-gray-200">{streaming.content}</p>
        ) : null}
        {waiting && <p className="text-gray-500">Thinking…</p>}
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
