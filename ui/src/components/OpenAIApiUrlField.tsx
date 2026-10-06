'use client';

import { ChangeEvent } from 'react';

interface Props {
  value: string;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}

export default function OpenAIApiUrlField({ value, onChange }: Props) {
  return (
    <div>
      <label htmlFor="OPENAI_API_URL" className="block text-sm font-medium mb-2">
        OpenAI Compatible API URL
        <div className="text-gray-500 text-sm ml-1">
          Base URL of an OpenAI-compatible API, for example http://localhost:1234/v1.
        </div>
      </label>
      <input
        type="text"
        id="OPENAI_API_URL"
        name="OPENAI_API_URL"
        value={value}
        onChange={onChange}
        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
        placeholder="http://localhost:1234/v1"
      />
    </div>
  );
}
