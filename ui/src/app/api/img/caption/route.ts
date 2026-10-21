import { NextResponse } from 'next/server';
import fs from 'fs';
import { isPathInsideRoot } from '@/server/safePath';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, caption } = body;
    let datasetsPath = await getDatasetsRoot();
    // make sure the dataset path is in the image path
    if (!isPathInsideRoot(imgPath, datasetsPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    // check for caption
    const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
    // save caption to file
    fs.writeFileSync(captionPath, caption);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to save caption:', error);
    return NextResponse.json({ error: 'Failed to save caption' }, { status: 500 });
  }
}
