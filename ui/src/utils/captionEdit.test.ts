import { describe, expect, it } from 'vitest';
import {
  CAPTION_EMPTY,
  CAPTION_NOT_LOADED,
  CAPTION_NOTHING_TO_UNDO,
  CAPTION_SAVE_FAILED,
  CAPTION_UNSAVED,
  CaptionEditState,
  isUndoAvailable,
  runCaptionEdit,
  runCaptionUndo,
} from '@/utils/captionEdit';

function state(overrides: Partial<CaptionEditState> = {}): CaptionEditState {
  return {
    caption: 'dog',
    savedCaption: 'dog',
    loaded: true,
    slot: null,
    ...overrides,
  };
}

describe('runCaptionEdit', () => {
  it('replaces the caption and remembers the previous one', async () => {
    const saved: string[] = [];
    const result = await runCaptionEdit(state(), '  brown dog  ', async caption => {
      saved.push(caption);
    });
    expect(saved).toEqual(['brown dog']);
    expect(result).toEqual({
      ok: true,
      state: {
        caption: 'brown dog',
        savedCaption: 'brown dog',
        loaded: true,
        slot: { previous: 'dog', applied: 'brown dog' },
      },
    });
  });

  it('replaces the undo slot on a second edit', async () => {
    const first = await runCaptionEdit(state(), 'brown dog', async () => {});
    if (!first.ok) throw new Error(first.message);
    const second = await runCaptionEdit(first.state, 'red dog', async () => {});
    expect(second).toMatchObject({
      ok: true,
      state: { savedCaption: 'red dog', slot: { previous: 'brown dog', applied: 'red dog' } },
    });
  });

  it('rejects an edit while the field has unsaved text', async () => {
    let writes = 0;
    const result = await runCaptionEdit(state({ caption: 'typed' }), 'brown dog', async () => {
      writes += 1;
    });
    expect(writes).toBe(0);
    expect(result).toMatchObject({ ok: false, message: CAPTION_UNSAVED, state: state({ caption: 'typed' }) });
  });

  it('rejects an edit before the caption has loaded', async () => {
    const result = await runCaptionEdit(state({ loaded: false, caption: '', savedCaption: '' }), 'brown dog', async () => {});
    expect(result).toMatchObject({ ok: false, message: CAPTION_NOT_LOADED });
  });

  it('rejects an empty caption', async () => {
    const result = await runCaptionEdit(state(), '   ', async () => {});
    expect(result).toMatchObject({ ok: false, message: CAPTION_EMPTY });
  });

  it('leaves state unchanged when the save fails', async () => {
    const current = state();
    const result = await runCaptionEdit(current, 'brown dog', async () => {
      throw new Error('disk full');
    });
    expect(result).toEqual({ ok: false, message: CAPTION_SAVE_FAILED, state: current });
  });
});

describe('runCaptionUndo', () => {
  it('restores the previous caption, including an empty one', async () => {
    const edited = await runCaptionEdit(state({ caption: '', savedCaption: '' }), 'brown dog', async () => {});
    if (!edited.ok) throw new Error(edited.message);
    const saved: string[] = [];
    const undone = await runCaptionUndo(edited.state, async caption => {
      saved.push(caption);
    });
    expect(saved).toEqual(['']);
    expect(undone).toMatchObject({
      ok: true,
      state: { caption: '', savedCaption: '', slot: null },
    });
    if (undone.ok) expect(isUndoAvailable(undone.state)).toBe(false);
  });

  it('hides undo after a later manual save', async () => {
    const edited = await runCaptionEdit(state(), 'brown dog', async () => {});
    if (!edited.ok) throw new Error(edited.message);
    const manual = { ...edited.state, caption: 'typed by hand', savedCaption: 'typed by hand' };
    expect(isUndoAvailable(manual)).toBe(false);
    const result = await runCaptionUndo(manual, async () => {});
    expect(result).toMatchObject({ ok: false, message: CAPTION_NOTHING_TO_UNDO, state: manual });
  });

  it('hides undo while the field is dirty', async () => {
    const edited = await runCaptionEdit(state(), 'brown dog', async () => {});
    if (!edited.ok) throw new Error(edited.message);
    const dirty = { ...edited.state, caption: 'brown dog plus' };
    expect(isUndoAvailable(dirty)).toBe(false);
  });

  it('leaves state unchanged when undoing fails to save', async () => {
    const edited = await runCaptionEdit(state(), 'brown dog', async () => {});
    if (!edited.ok) throw new Error(edited.message);
    const result = await runCaptionUndo(edited.state, async () => {
      throw new Error('read only');
    });
    expect(result).toEqual({ ok: false, message: CAPTION_SAVE_FAILED, state: edited.state });
  });
});
