import { describe, expect, it } from 'vitest';
import { assistantTurnContent, parseAgentChatStreamLine } from '@/utils/agentChatStream';

describe('parseAgentChatStreamLine', () => {
  it('accepts an edit event', () => {
    expect(parseAgentChatStreamLine('{"type":"edit","caption":"brown dog"}')).toEqual({
      type: 'edit',
      caption: 'brown dog',
    });
  });

  it('accepts reasoning, content, and error lines', () => {
    expect(parseAgentChatStreamLine('{"type":"content","delta":"ok"}')).toEqual({ type: 'content', delta: 'ok' });
    expect(parseAgentChatStreamLine('{"type":"reasoning","delta":"..."}')).toEqual({ type: 'reasoning', delta: '...' });
    expect(parseAgentChatStreamLine('{"type":"error","message":"nope"}')).toEqual({ type: 'error', message: 'nope' });
  });

  it('rejects a malformed edit and other junk', () => {
    expect(parseAgentChatStreamLine('{"type":"edit"}')).toBeNull();
    expect(parseAgentChatStreamLine('{"type":"edit","caption":1}')).toBeNull();
    expect(parseAgentChatStreamLine('not json')).toBeNull();
    expect(parseAgentChatStreamLine('{"type":"tool_call"}')).toBeNull();
  });
});

describe('assistantTurnContent', () => {
  it('returns the reply unchanged when there is no edit', () => {
    expect(assistantTurnContent('hello')).toBe('hello');
  });

  it('stores only the note when the reply is empty', () => {
    expect(assistantTurnContent('  ', 'brown dog')).toBe('Caption updated to:\nbrown dog');
  });

  it('appends the note after a text reply', () => {
    expect(assistantTurnContent('Updated.', 'brown dog')).toBe('Updated.\n\nCaption updated to:\nbrown dog');
  });

  it('drops blank lines the reply already ended with', () => {
    expect(assistantTurnContent('Updated.\n\n', 'brown dog')).toBe('Updated.\n\nCaption updated to:\nbrown dog');
  });
});
