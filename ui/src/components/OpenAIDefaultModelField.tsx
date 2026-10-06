'use client';

import { ChangeEvent } from 'react';

interface Props {
  value: string;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}

export default function OpenAIDefaultModelField({ value, onChange }: Props) {
  return (
    <div>
      <label htmlFor="OPENAI_DEFAULT_MODEL" className="block text-sm font-medium mb-2">
        Default model
        <div className="text-gray-500 text-sm ml-1">Used only when the API returns no models.</div>
      </label>
      <input
        type="text"
        id="OPENAI_DEFAULT_MODEL"
        name="OPENAI_DEFAULT_MODEL"
        value={value}
        onChange={onChange}
        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
        placeholder="model-id"
      />
    </div>
  );
}
