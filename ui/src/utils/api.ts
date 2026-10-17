import axios from 'axios';
import { createGlobalState } from 'react-global-hooks';

const AUTH_TOKEN_KEY = 'AI_TOOLKIT_AUTH';

export const isAuthorizedState = createGlobalState(false);

export function authHeaders(): { Authorization: string } | Record<string, never> {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

export function clearUnauthorized(status: number | undefined): void {
  if (status !== 401) return;
  localStorage.removeItem(AUTH_TOKEN_KEY);
  isAuthorizedState.set(false);
}

export const apiClient = axios.create();

export async function fetchCaption(imgPath: string): Promise<string> {
  const res = await apiClient.post<string>('/api/caption/get', { imgPath }, { responseType: 'text' });
  return typeof res.data === 'string' ? res.data : '';
}

// Add a request interceptor to add token from localStorage
apiClient.interceptors.request.use(config => {
  const headers = authHeaders();
  if ('Authorization' in headers) {
    config.headers['Authorization'] = headers.Authorization;
  }
  return config;
});

// Add a response interceptor to handle 401 errors
apiClient.interceptors.response.use(
  response => response, // Return successful responses as-is
  error => {
    const status = typeof error?.response?.status === 'number' ? error.response.status : undefined;
    clearUnauthorized(status);

    // Reject the promise with the error so calling code can still catch it
    return Promise.reject(error);
  },
);
