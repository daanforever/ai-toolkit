'use client';

import type { ReactNode } from 'react';
import type { Components } from 'react-markdown';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Props {
  children: string;
  className?: string;
}

function hasClass(className: string | undefined, name: string): boolean {
  return className?.split(/\s+/).includes(name) ?? false;
}

const components: Components = {
  p: ({ children }) => <p className="break-words">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  del: ({ children }) => <del className="line-through">{children}</del>,
  h1: ({ children }) => <Heading tag="h1">{children}</Heading>,
  h2: ({ children }) => <Heading tag="h2">{children}</Heading>,
  h3: ({ children }) => <Heading tag="h3">{children}</Heading>,
  h4: ({ children }) => <Heading tag="h4">{children}</Heading>,
  h5: ({ children }) => <Heading tag="h5">{children}</Heading>,
  h6: ({ children }) => <Heading tag="h6">{children}</Heading>,
  ul: ({ children, className }) => (
    <ul className={hasClass(className, 'contains-task-list') ? 'list-none pl-0' : 'list-disc pl-5'}>{children}</ul>
  ),
  ol: ({ children }) => <ol className="list-decimal pl-5">{children}</ol>,
  li: ({ children, className }) => (
    <li className={hasClass(className, 'task-list-item') ? 'list-none' : undefined}>{children}</li>
  ),
  a: ({ children, href }) =>
    href ? (
      <a href={href} target="_blank" rel="noopener noreferrer" className="break-words underline">
        {children}
      </a>
    ) : (
      <span>{children}</span>
    ),
  code: ({ children, className }) => (
    <code className={['rounded bg-gray-800 px-1 font-mono text-[0.85em]', className].filter(Boolean).join(' ')}>
      {children}
    </code>
  ),
  pre: ({ children }) => (
    <pre className="max-w-full overflow-x-auto rounded bg-gray-900 p-2 font-mono text-[0.85em] [&_code]:block [&_code]:whitespace-pre [&_code]:rounded-none [&_code]:bg-transparent [&_code]:p-0">
      {children}
    </pre>
  ),
  blockquote: ({ children }) => <blockquote className="border-l-2 border-gray-600 pl-3">{children}</blockquote>,
  hr: () => <hr className="border-gray-700" />,
  table: ({ children }) => (
    <div className="max-w-full overflow-x-auto">
      <table className="w-max min-w-full border-collapse text-left">{children}</table>
    </div>
  ),
  th: ({ children, style }) => (
    <th style={style} className="border border-gray-600 px-2 py-1 font-semibold">
      {children}
    </th>
  ),
  td: ({ children, style }) => (
    <td style={style} className="border border-gray-600 px-2 py-1">
      {children}
    </td>
  ),
  input: ({ checked, type }) =>
    type === 'checkbox' ? (
      <input type="checkbox" checked={Boolean(checked)} disabled className="mr-1 align-middle" />
    ) : null,
};

function Heading({ tag, children }: { tag: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6'; children: ReactNode }) {
  const Tag = tag;
  const size = tag === 'h1' ? 'text-base' : 'text-sm';
  return <Tag className={`${size} font-semibold`}>{children}</Tag>;
}

export default function MarkdownText({ children, className }: Props) {
  return (
    <div className={['min-w-0 max-w-full break-words text-sm [&>*+*]:mt-2', className ?? 'text-gray-200'].join(' ')}>
      <Markdown remarkPlugins={[remarkGfm]} components={components} disallowedElements={['img']}>
        {children}
      </Markdown>
    </div>
  );
}
