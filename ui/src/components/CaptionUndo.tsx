'use client';

import { MouseEvent } from 'react';

interface Props {
  disabled?: boolean;
  onUndo: () => void;
}

export default function CaptionUndo({ disabled = false, onUndo }: Props) {
  const onClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    onUndo();
  };

  return (
    <p className="text-sm text-gray-400">
      Caption updated.{' '}
      <button
        type="button"
        aria-label="Undo caption"
        disabled={disabled}
        onClick={onClick}
        className="underline disabled:cursor-not-allowed disabled:opacity-40"
      >
        Undo
      </button>
    </p>
  );
}
