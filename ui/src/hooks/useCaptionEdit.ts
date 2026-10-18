'use client';

import { useEffect, useRef, useState } from 'react';
import {
  CAPTION_IMAGE_CHANGED,
  CAPTION_NOT_LOADED,
  CaptionEditResult,
  CaptionEditState,
  CaptionUndoSlot,
  isUndoAvailable,
  runCaptionEdit,
  runCaptionUndo,
} from '@/utils/captionEdit';
import { persistCaption } from '@/utils/persistCaption';

interface Args {
  imgPath: string | null;
  caption: string;
  savedCaption: string;
  isCaptionLoaded: boolean;
  setCaption: (value: string) => void;
  setSavedCaption: (value: string) => void;
  onCaptionSaved?: (saved: { imgPath: string; caption: string }) => void;
}

export function useCaptionEdit({
  imgPath,
  caption,
  savedCaption,
  isCaptionLoaded,
  setCaption,
  setSavedCaption,
  onCaptionSaved,
}: Args) {
  const [slot, setSlot] = useState<CaptionUndoSlot | null>(null);
  const imgPathRef = useRef(imgPath);
  const captionRef = useRef(caption);
  const savedRef = useRef(savedCaption);
  const loadedRef = useRef(isCaptionLoaded);
  const slotRef = useRef(slot);
  const onCaptionSavedRef = useRef(onCaptionSaved);
  imgPathRef.current = imgPath;
  captionRef.current = caption;
  savedRef.current = savedCaption;
  loadedRef.current = isCaptionLoaded;
  slotRef.current = slot;
  onCaptionSavedRef.current = onCaptionSaved;

  useEffect(() => {
    setSlot(null);
  }, [imgPath]);

  const snapshot = (): CaptionEditState => ({
    caption: captionRef.current,
    savedCaption: savedRef.current,
    loaded: loadedRef.current,
    slot: slotRef.current,
  });

  const applyState = (path: string, state: CaptionEditState) => {
    setCaption(state.caption);
    setSavedCaption(state.savedCaption);
    setSlot(state.slot);
    onCaptionSavedRef.current?.({ imgPath: path, caption: state.savedCaption });
  };

  const applyEdit = async (next: string): Promise<CaptionEditResult> => {
    const path = imgPathRef.current;
    if (!path) return { ok: false, message: CAPTION_NOT_LOADED };
    const result = await runCaptionEdit(snapshot(), next, captionText => persistCaption(path, captionText));
    if (imgPathRef.current !== path) return { ok: false, message: CAPTION_IMAGE_CHANGED };
    if (!result.ok) return { ok: false, message: result.message };
    applyState(path, result.state);
    return { ok: true, caption: result.state.savedCaption };
  };

  const undo = async (): Promise<CaptionEditResult> => {
    const path = imgPathRef.current;
    if (!path) return { ok: false, message: CAPTION_NOT_LOADED };
    const result = await runCaptionUndo(snapshot(), captionText => persistCaption(path, captionText));
    if (imgPathRef.current !== path) return { ok: false, message: CAPTION_IMAGE_CHANGED };
    if (!result.ok) return { ok: false, message: result.message };
    applyState(path, result.state);
    return { ok: true, caption: result.state.savedCaption };
  };

  const state = snapshot();
  return { applyEdit, undo, undoAvailable: isUndoAvailable(state) };
}
