import { apiClient } from '@/utils/api';

export async function persistCaption(imgPath: string, caption: string): Promise<void> {
  await apiClient.post('/api/img/caption', { imgPath, caption });
}
