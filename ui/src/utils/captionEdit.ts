export const CAPTION_NOT_LOADED = 'Caption is not loaded';
export const CAPTION_UNSAVED = 'Caption has unsaved edits';
export const CAPTION_SAVE_FAILED = 'Failed to save caption';
export const CAPTION_IMAGE_CHANGED = 'Image changed';
export const CAPTION_EMPTY = 'Caption is empty';
export const CAPTION_NOTHING_TO_UNDO = 'Nothing to undo';

export interface CaptionUndoSlot {
  previous: string;
  applied: string;
}

export interface CaptionEditState {
  caption: string;
  savedCaption: string;
  loaded: boolean;
  slot: CaptionUndoSlot | null;
}

export type CaptionEditResult = { ok: true; caption: string } | { ok: false; message: string };

type CaptionEditRun =
  | { ok: true; state: CaptionEditState }
  | { ok: false; message: string; state: CaptionEditState };

export function isUndoAvailable(state: CaptionEditState): boolean {
  const slot = state.slot;
  if (!slot) return false;
  if (slot.previous === slot.applied) return false;
  if (state.savedCaption !== slot.applied) return false;
  return state.caption.trim() === state.savedCaption;
}

export function previewCaptionEdit(state: CaptionEditState, next: string): { state: CaptionEditState } | { error: string } {
  if (!state.loaded) return { error: CAPTION_NOT_LOADED };
  if (state.caption.trim() !== state.savedCaption) return { error: CAPTION_UNSAVED };
  const applied = next.trim();
  if (applied === '') return { error: CAPTION_EMPTY };
  return {
    state: {
      caption: applied,
      savedCaption: applied,
      loaded: true,
      slot: { previous: state.savedCaption, applied },
    },
  };
}

export function previewCaptionUndo(state: CaptionEditState): { state: CaptionEditState } | { error: string } {
  if (!isUndoAvailable(state) || !state.slot) return { error: CAPTION_NOTHING_TO_UNDO };
  const restored = state.slot.previous;
  return {
    state: {
      caption: restored,
      savedCaption: restored,
      loaded: state.loaded,
      slot: null,
    },
  };
}

async function commit(
  state: CaptionEditState,
  preview: { state: CaptionEditState } | { error: string },
  persist: (caption: string) => Promise<void>,
): Promise<CaptionEditRun> {
  if ('error' in preview) return { ok: false, message: preview.error, state };
  try {
    await persist(preview.state.savedCaption);
  } catch (error) {
    console.error('Error saving caption:', error);
    return { ok: false, message: CAPTION_SAVE_FAILED, state };
  }
  return { ok: true, state: preview.state };
}

export function runCaptionEdit(
  state: CaptionEditState,
  next: string,
  persist: (caption: string) => Promise<void>,
): Promise<CaptionEditRun> {
  return commit(state, previewCaptionEdit(state, next), persist);
}

export function runCaptionUndo(
  state: CaptionEditState,
  persist: (caption: string) => Promise<void>,
): Promise<CaptionEditRun> {
  return commit(state, previewCaptionUndo(state), persist);
}
