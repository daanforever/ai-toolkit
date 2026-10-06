'use client';

import { FormEvent } from 'react';
import { Send } from 'lucide-react';

export default function AgentChatPanel() {
  const blockSubmit = (event: FormEvent) => {
    event.preventDefault();
  };

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-l border-gray-700 bg-gray-950">
      <div className="flex flex-1 items-center justify-center overflow-y-auto px-4 text-sm text-gray-500">
        AI agent — coming soon
      </div>
      <form className="shrink-0 border-t border-gray-700 p-3" onSubmit={blockSubmit}>
        <div className="relative">
          <textarea
            aria-label="Message"
            rows={3}
            placeholder="Message"
            className="w-full resize-none rounded-lg border border-gray-700 bg-gray-900 py-2 pl-3 pr-10 text-sm text-gray-200 outline-none focus:ring-2 focus:ring-gray-600"
          />
          <button
            type="submit"
            disabled
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
